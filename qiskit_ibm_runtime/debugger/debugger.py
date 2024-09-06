# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A debugger."""

from __future__ import annotations
from typing import Optional, Sequence, Type, Union

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qiskit.transpiler.passmanager import PassManager
from qiskit.primitives.containers import EstimatorPubLike, PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import BackendV2 as Backend

from qiskit_ibm_runtime.debugger.figure_or_merit import FOM, Ratio
from qiskit_ibm_runtime.transpiler.passes.cliffordization import ConvertISAToClifford
from qiskit_ibm_runtime.utils import validate_estimator_pubs, validate_isa_circuits


def _get_result(
    coerced_pubs: Sequence[EstimatorPub],
    result: Union[str, PrimitiveResult],
    noise_model: NoiseModel,
    default_precision: float,
    seed_simulator: Union[int, None],
):
    r"""Retrieves the results for a given debugger mode."""

    if isinstance(result, PrimitiveResult):
        return result

    backend_options = {"method": "stabilizer", "seed_simulator": seed_simulator}
    options = {"backend_options": backend_options, "default_precision": default_precision}

    if result == "ideal_sim":
        estimator = AerEstimator(options=options)
        return estimator.run(coerced_pubs).result()
    if result == "noisy_sim":
        options["backend_options"]["noise_model"] = noise_model
        estimator = AerEstimator(options=options)
        return estimator.run(coerced_pubs).result()
    raise ValueError("Cannot retrieve the results.")


def _validate_pubs(backend: Backend, pubs: Sequence[EstimatorPub], validate_clifford=True):
    r"""Validates a list PUBs by running the :meth:`.~validate_estimator_pubs` and
    :meth:`.~validate_isa_circuits` methods, and optionally, by checking if the PUBs
    are Clifford.
    """
    validate_estimator_pubs(pubs)
    validate_isa_circuits([pub.circuit for pub in pubs], backend.target)

    if validate_clifford:
        for pub in pubs:
            cliff_circ = PassManager([ConvertISAToClifford()]).run(pub.circuit)
            if pub.circuit != cliff_circ:
                raise ValueError(
                    "Given ``pubs`` contain a non-Clifford circuit. To fix, consider using the "
                    "``ConvertISAToClifford`` pass to map your circuits to the nearest Clifford"
                    " circuits, then try again."
                )


class Debugger:
    r"""A class that users of the Estimator primitive can use to understand the expected
    performance of their queries.

    Args:
        backend: A backend.
        noise_model: A noise model for the operations of the given backend. If ``None``, it
            defaults to the noise model generated by :meth:`NoiseModel.from_backend`.
    """

    def __init__(self, backend: Backend, noise_model: Optional[NoiseModel] = None) -> None:
        self._backend = backend
        self._noise_model = noise_model or NoiseModel.from_backend(
            backend, thermal_relaxation=False
        )

    @property
    def backend(self) -> Backend:
        r"""
        The backend used by this debugger.
        """
        return self._backend

    @property
    def noise_model(self) -> NoiseModel:
        r"""
        The noise model used by this debugger for the noisy simulations.
        """
        return self._noise_model

    def compare(
        self,
        pubs: Sequence[EstimatorPubLike],
        result1: Union[str, PrimitiveResult] = "noisy_sim",
        result2: Union[str, PrimitiveResult] = "ideal_sim",
        fom: Type[FOM] = Ratio,
        default_precision: float = 0,
        seed_simulator: Optional[int] = None,
    ):
        r"""
        Compute figures of merit that can help understand the performance of an estimation task
        prior to executing the task on a real backend.

        Given a target estimation task, it compares different results obtained by performing an
        ideal simulation of the target task, by implementing a noisy simulation of it, or by
        running it on an actual backend.

        Here are a few notable examples that illustrate how this function can be useful:

            * With default values of ``result1`` (``"noisy_sim"``), ``result2`` (``"ideal_sim"``),
              and ``fom`` (:class:`~.Ratio`), it classically simulates the estimation task
              both in the presence and in the absence of noise, and it returns the signal-to-noise
              ratio between noisy and ideal results, a quantity that can help predicting the
              performance of error mitigation.

            * Setting ``result1`` to ``"noisy_sim"`` and providing experimental results for
              ``result2``, it compares the experimental results with the results of a noisy simulation.
              This can help understand if a given noise model is a good approximation of the noise
              processes that affect the backend in use.

            * Setting ``result1`` to ``"ideal_sim"`` and providing experimental results for
              ``result2``, it compares the experimental results with the results of an ideal simulation.
              This can help understand how well a backend can perform a certain estimation task, or
              (if error mitigation was used to obtain the experimental results) the performance of
              different mitigation strategies.

        .. note::
            To ensure scalability, every circuit in ``pubs`` is required to be a Clifford circuit,
            so that it can be simulated efficiently regardless of its size. For estimation tasks
            that involve non-Clifford circuits, the recommended workflow consists of mapping
            the non-Clifford circuits to the nearest Clifford circuits using the
            :class:`.~ConvertISAToClifford` transpiler pass, or equivalently, to use the debugger's
            :meth:`to_clifford` convenience method.

        .. note::
            This function assumes, but does not check, that any experimental result provided as an
            input was previously obtained by running the given ``pubs`` on a backend.

        Args:
            pubs: The PUBs specifying the estimation task of interest.
            result1: The first set of :class:`.~PrimitiveResult`\\s to use in the comparison, or
                alternatively a string (allowed values are ``ideal_sim`` and ``noisy_sim``). If
                a string is passed, the results are produced internally by running a classical
                simulation.
            result2: The second set of :class:`.~PrimitiveResult`\\s to use in the comparison, or
                alternatively a string (allowed values are ``ideal_sim`` and ``noisy_sim``). If
                a string is passed, the results are produced internally by running a classical
                simulation.
            fom: The figure of merit use to compare ``result1`` and ``result2``. Defaults to
                computing the ratio.
            default_precision: The default precision used to run the ideal and noisy simulations.
            seed_simulator: A seed for the simulator.
        """
        for result in [result1, result2]:
            if isinstance(result, str) and result not in ["ideal_sim", "noisy_sim", "exp"]:
                raise ValueError(f"Invalid result '{result}', must be 'ideal_sim' or 'noisy_sim'.")
        _validate_pubs(self.backend, coerced_pubs := [EstimatorPub.coerce(pub) for pub in pubs])

        r1 = _get_result(coerced_pubs, result1, self.noise_model, default_precision, seed_simulator)
        r2 = _get_result(coerced_pubs, result2, self.noise_model, default_precision, seed_simulator)
        return fom(r1, r2)

    def to_clifford(self, pubs: Sequence[EstimatorPubLike]) -> list[EstimatorPub]:
        r"""
        A convenience method that returns the cliffordized version of the given ``pubs``, obtained
        by run the :class:`.~ConvertISAToClifford` transpiler pass on the PUBs' circuits.

        Args:
            pubs: The PUBs to turn into Clifford PUBs.
        Returns:
            The Clifford PUBs.
        """
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        _validate_pubs(self.backend, coerced_pubs, False)

        ret = []
        for pub in coerced_pubs:
            new_pub = EstimatorPub(
                PassManager([ConvertISAToClifford()]).run(pub.circuit),
                pub.observables,
                pub.parameter_values,
                pub.precision,
                False,
            )
            ret.append(new_pub)

        return ret

    def __repr__(self) -> str:
        return f'Debugger(backend="{self.backend.name}")'
