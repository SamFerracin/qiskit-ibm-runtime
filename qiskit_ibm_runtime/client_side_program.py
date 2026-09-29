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

"""Client-side Estimator primitive."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .base_primitive import get_mode_service_backend
from .executor import Executor

if TYPE_CHECKING:
    from qiskit.providers import BackendV2

    from .batch import Batch
    from .fake_provider.local_runtime_job import LocalRuntimeJob
    from .options_models import ExecutorOptions
    from .options_models.base import BaseOptionsModel
    from .quantum_program import QuantumProgram
    from .runtime_job_v2 import RuntimeJobV2
    from .session import Session

InputT = TypeVar("InputT")
OptionsT = TypeVar("OptionsT", bound="BaseOptionsModel")

logger = logging.getLogger(__name__)


class ClientSideProgram(ABC, Generic[InputT, OptionsT]):
    """Base class for all client-side programs.

    Args:
        mode: The execution mode used to make the primitive query. It can be:

            * A :class:`~qiskit.providers.BackendV2` if you are using job mode.
            * A :class:`~qiskit_ibm_runtime.Session` if you are using session execution mode.
            * A :class:`~qiskit_ibm_runtime.Batch` if you are using batch execution mode.

            Refer to the `IBM Quantum Compute documentation
            <https://quantum.cloud.ibm.com/docs/guides/execution-modes>`__
            for more information about execution modes.

        options: The options of this program.
    """

    options: OptionsT
    """The options of this program."""

    @property
    @abstractmethod
    def _semantic_role(self) -> str:
        """Semantic role indicating how execution results should be post-processed."""

    @property
    @abstractmethod
    def default_options(self) -> OptionsT:
        """The default options of this program."""

    def __init__(
        self,
        mode: BackendV2 | Session | Batch | None = None,
        options: OptionsT | dict | None = None,
    ):
        super().__init__()

        self._mode, self._service, self._backend = get_mode_service_backend(mode)
        self.options = options if options is not None else self.default_options  # type: ignore[assignment]

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute ``name`` to ``value``.

        Handle ``options`` as a special case, ensuring it is set to an ``EstimatorOptions``
        instance. This is an alternative to using ``@setter``, as the setter causes issues in
        ``ipython`` autocomplete features.
        """
        if name == "options":
            if isinstance(value, dict):
                value = self.options.update(**value)
            elif not isinstance(value, type(self.options)):
                raise TypeError(f"Expected {type(self.options)} or dict, got {type(value)}")

        super().__setattr__(name, value)

    def backend(self) -> BackendV2:
        """Return the backend the primitive query will be run on."""
        return self._backend

    @property
    def mode(self) -> Session | Batch | None:
        """Return the execution mode used by this primitive.

        Returns:
            Mode used by this primitive, or ``None`` if an execution mode is not used.
        """
        return self._mode

    @abstractmethod
    def prepare(self, input: InputT) -> tuple[QuantumProgram, ExecutorOptions]:
        """The function used to map this program's inputs to the inputs of Executor."""

    def _run(self, input: InputT, *, dry_run: bool = False) -> RuntimeJobV2 | LocalRuntimeJob:
        """Run a job via Executor."""
        # Pre-process: Convert Estimator input into a QuantumProgram
        logger.info("Starting pre-processing")
        quantum_program, executor_options = self.prepare(input)

        # Set semantic role for post-processing dispatch
        quantum_program._semantic_role = self._semantic_role

        executor = Executor(mode=self._mode or self._backend, options=executor_options)

        logger.info(
            "Submitting %d item%s to executor with %d total shots",
            len(quantum_program.items),
            "s" if len(quantum_program.items) > 1 else "",
            quantum_program.shots * sum(item.size() for item in quantum_program.items),
        )
        return executor.run(quantum_program, dry_run=dry_run)
