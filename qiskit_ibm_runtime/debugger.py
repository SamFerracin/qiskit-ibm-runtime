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

"""Estimator primitive."""

from __future__ import annotations
import numpy as np
import os
from typing import Optional, Dict, Sequence, Any, Union, Iterable
import logging

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1, BackendV2
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from .runtime_job import RuntimeJob
from .runtime_job_v2 import RuntimeJobV2
from .ibm_backend import IBMBackend
from .options.estimator_options import EstimatorOptions
from .base_primitive import BasePrimitiveV1, BasePrimitiveV2
from .utils.deprecation import deprecate_arguments, issue_deprecation_msg
from .utils.qctrl import validate as qctrl_validate
from .utils.qctrl import validate_v2 as qctrl_validate_v2
from .utils import validate_estimator_pubs, validate_isa_circuits

# pylint: disable=unused-import,cyclic-import
from .session import Session
from .batch import Batch

logger = logging.getLogger(__name__)


class Debugger:
    r"""A debugger."""

    def __init__(self, backend: Union[BackendV1, BackendV2]) -> None:
        self._backend = backend

    @property
    def backend(self):
        return self._backend

    def run(self, pubs: Iterable[EstimatorPubLike]):
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]

        # validation
        validate_estimator_pubs(coerced_pubs)
        for pub in coerced_pubs:
            validate_isa_circuits([pub.circuit], self.backend.target)

        # ideal simulation
        options = {"method": "stabilizer"}
        ideal_estimator = AerEstimator(options={"backend_options": options})
        ideal_results = ideal_estimator.run(coerced_pubs).result()

        # noisy simulation
        noise_model = NoiseModel.from_backend(self.backend, thermal_relaxation=False)
        options.update({"noise_model": noise_model})
        noisy_estimator = AerEstimator(options={"backend_options": options})
        noisy_results = noisy_estimator.run(coerced_pubs).result()

        ret = []
        for noisy_result, ideal_result in zip(noisy_results, ideal_results):
            noisy_evs = noisy_result.data.evs
            ideal_evs = ideal_result.data.evs
            ret.append(
                np.divide(
                    noisy_evs,
                    ideal_evs,
                    out=np.zeros_like(noisy_evs, dtype=float),
                    where=ideal_evs != 0,
                )
            )

        return ret

    def __repr__(self) -> str:
        return f"Debugger({self.backend.name})"
