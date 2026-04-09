# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Decoders for quantum programs."""

from __future__ import annotations

import logging

from ibm_quantum_schemas.executor.version_0_1 import ParamsModel as ParamsModel_0_1
from ibm_quantum_schemas.executor.version_0_2 import ParamsModel as ParamsModel_0_2

from .converters import quantum_program_from_0_1
from .converters import quantum_program_from_0_2

from ..quantum_program import QuantumProgram
from ..options.executor_options import ExecutorOptions

logger = logging.getLogger(__name__)

AVAILABLE_DECODERS = {
    "v0.1": (quantum_program_from_0_1, ParamsModel_0_1),
    "v0.2": (quantum_program_from_0_2, ParamsModel_0_2),
}


class QuantumProgramDecoder:
    """Decoder for quantum programs."""

    def get_supported_schema_versions(self) -> set[str]:
        return set(AVAILABLE_DECODERS)

    def decode(self, raw_program: dict[str, str]) -> tuple[QuantumProgram, ExecutorOptions]:
        """Decode raw json to result type."""
        try:
            schema_version = raw_program["schema_version"]
        except KeyError:
            raise ValueError("Missing schema version.")

        try:
            decoder, model = AVAILABLE_DECODERS[schema_version]
        except KeyError:
            raise ValueError(f"No decoder found for schema version {schema_version}.")

        return decoder(model.model_validate_json(raw_program))
