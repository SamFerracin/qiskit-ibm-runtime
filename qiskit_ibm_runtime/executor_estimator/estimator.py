# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Executor-based EstimatorV2 primitive."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from ..exceptions import IBMInputValueError
from ..executor import Executor
from ..executor.dynamical_decoupling import apply_dynamical_decoupling
from ..options_models.estimator_options import EstimatorOptions
from .pec.prepare_pec import prepare_pec
from .prepare import prepare
from .utils import resolve_precision

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit.primitives.containers.estimator_pub import EstimatorPubLike
    from qiskit.providers import BackendV2

    from ..batch import Batch
    from ..runtime_job_v2 import RuntimeJobV2
    from ..session import Session

logger = logging.getLogger(__name__)


class EstimatorV2(BaseEstimatorV2):
    """Executor-based EstimatorV2 primitive for Qiskit Runtime.

    This is an implementation of EstimatorV2 built on top of the Executor primitive,
    enabling transparent client-side processing with faster feedback loops and greater
    user control.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import SparsePauliOp
            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit_ibm_runtime.executor_estimator import EstimatorV2

            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)

            # Create a simple circuit
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            # Define observable
            observable = SparsePauliOp.from_list([("ZZ", 1), ("XX", 1)])

            # Run the estimator with options
            estimator = EstimatorV2(mode=backend)
            estimator.options.default_precision = 0.01
            estimator.options.execution.init_qubits = True
            job = estimator.run([(circuit, observable)])
            result = job.result()

    Args:
        mode: The execution mode used to make the primitive query. It can be:

            * A :class:`~qiskit.providers.BackendV2` if you are using job mode.
            * A :class:`~qiskit_ibm_runtime.Session` if you are using session execution mode.
            * A :class:`~qiskit_ibm_runtime.Batch` if you are using batch execution mode.

            Refer to the `Qiskit Runtime documentation
            <https://quantum.cloud.ibm.com/docs/guides/execution-modes>`_
            for more information about execution modes.

        options: Estimator options.
            See
            :class:`~qiskit_ibm_runtime.options_models.estimator_options.EstimatorOptions`
            for all available options.
    """

    options: EstimatorOptions
    """The options of this Estimator."""

    def __init__(
        self,
        mode: BackendV2 | Session | Batch | None = None,
        options: EstimatorOptions | dict | None = None,
    ):
        super().__init__()

        self._executor = Executor(mode=mode)

        # Coerced to `SampEstimatorOptionslerOptions` via `__setattr__()`.
        self.options = options if options is not None else EstimatorOptions()  # type: ignore[assignment]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute ``name`` to ``value``.

        Handle ``options`` as a special case, ensuring it is set to an ``EstimatorOptions``
        instance. This is an alternative to using ``@setter``, as the setter causes issues in
        ``ipython`` autocomplete features.
        """
        if name == "options":
            if isinstance(value, dict):
                value = EstimatorOptions(**value)
            elif not isinstance(value, EstimatorOptions):
                raise TypeError(f"Expected EstimatorOptions or dict, got {type(value)}")

        super().__setattr__(name, value)

    def finalize_options(self) -> EstimatorOptions:
        """Finalize options."""
        # Begin by initializing options based on resilience level
        options = EstimatorOptions()
        if (resilience_level := self.options.resilience_level) == 0:
            options.twirling.enable_gates = False
            options.twirling.enable_measure = False
            options.resilience.measure_mitigation = False
        elif resilience_level == 1:
            options.twirling.enable_gates = False
            options.twirling.enable_measure = True
            options.resilience.measure_mitigation = True
        elif resilience_level == 2:
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
            options.resilience.measure_mitigation = True
            options.resilience.zne_mitigation = True
            raise ValueError("Not yet implemented -- missing ZNE.")

        # Dump user options, excluding values that have been set to ``None``--these values are
        # decided based on the resilience level.
        exclude = {
            "twirling": {
                field
                for field in ("enable_gates", "enable_measure")
                if getattr(self.options.twirling, field) is None
            },
            "resilience": {
                field
                for field in ("measure_mitigation", "zne_mitigation")
                if getattr(self.options.resilience, field) is None
            },
        }
        exclude = {k: v for k, v in exclude.items() if v}
        options.update(**self.options.model_dump(exclude=exclude))

        # Finally, force-set some values based on mitigation
        if options.resilience.measure_mitigation is True:
            options.twirling.enable_measure = True
        if options.resilience.zne_mitigation is True and options.resilience.zne.amplifier == "pea":
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True
        if options.resilience.pec_mitigation is True:
            options.twirling.enable_gates = True
            options.twirling.enable_measure = True

        return options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> RuntimeJobV2:
        """Submit a request to the estimator primitive.

        For moderate and complex workloads, the client-side processing done to map estimator inputs
        to executor inputs can be resource intensive and cause a delay between invoking the function
        and the ``job`` being submitted. In order to check the progress of the call, it is
        recommended to setup logging (with an ``INFO`` level) - see
        `Qiskit Runtime documentation
        <https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/runtime-service#logging>`_
        for more information.

        Args:
            pubs: An iterable of pub-like objects. For example, a list of circuits
                  and observables or tuples ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                       estimator pub that does not specify its own precision. If ``None``,
                       the value from ``options.default_precision`` will be used.

        Returns:
            The submitted job.

        Raises:
            ValueError: If backend is not provided.
            IBMInputValueError: If precision is not properly specified or if unsupported
                options are detected.
        """
        options = self.finalize_options()

        # Coerce pubs to EstimatorPub objects
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        # Convert pubs to QuantumProgram and map options using the selected prepare function
        logger.info("Starting pre-processing")

        resolved_precision = resolve_precision(coerced_pubs, precision)
        if resolved_precision is not None:
            shots = int(np.ceil(1.0 / (resolved_precision**2)))
        elif options.default_shots is not None:
            shots = int(options.default_shots)
        else:
            shots = int(np.ceil(1.0 / (options.default_precision**2)))

        if options.dynamical_decoupling.enable:
            for pub in coerced_pubs:
                if pub.circuit.has_control_flow_op():
                    raise IBMInputValueError(
                        "Dynamical decoupling is not compatible with dynamic circuits "
                        "(circuits with control flow operations)."
                    )

        # Route to appropriate prepare function
        if options.resilience.pec_mitigation:
            if options.resilience.noise_model_mapping is None:
                raise IBMInputValueError(
                    "When PEC mitigation is enabled, you must provide a noise model "
                    "via options.resilience.noise_model_mapping"
                )
            quantum_program = prepare_pec(
                pubs=coerced_pubs,
                twirling_options=options.twirling,
                shots=shots,
                pec_options=options.resilience.pec,
                noise_model_mapping=options.resilience.noise_model_mapping,
                measure_noise_learning=options.resilience.measure_noise_learning
                if options.resilience.measure_mitigation
                else None,
            )
        else:
            quantum_program = prepare(
                pubs=coerced_pubs,
                twirling_options=options.twirling,
                shots=shots,
                measure_noise_learning=options.resilience.measure_noise_learning
                if options.resilience.measure_mitigation
                else None,
            )

        if options.dynamical_decoupling.enable:
            quantum_program = apply_dynamical_decoupling(
                backend=self._executor._backend,
                dd_options=options.dynamical_decoupling,
                quantum_program=quantum_program,
            )
        # Serialize options (assuming passthrough is correctly configured)
        quantum_program.passthrough_data["post_processor"]["options"] = {  # type: ignore[index, call-overload]
            "twirling": options.twirling.model_dump(),
            "dynamical_decoupling": options.dynamical_decoupling.model_dump(),
            "resilience": options.resilience.model_dump(exclude={"noise_model_mapping"}),
        }

        executor_options = options.to_executor_options()

        # Set executor options
        self._executor.options = executor_options

        # Submit to executor
        logger.info(
            "Submitting %d pub%s to executor with %d shots",
            len(coerced_pubs),
            "s" if len(coerced_pubs) > 1 else "",
            quantum_program.shots,
        )

        return self._executor.run(quantum_program)
