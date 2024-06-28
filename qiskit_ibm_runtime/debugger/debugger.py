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
import numpy as np
from typing import Union, Iterable

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from qiskit.transpiler.passmanager import PassManager
from qiskit.primitives.containers import EstimatorPubLike
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import BackendV1, BackendV2

from qiskit_ibm_runtime.debugger.plugins import Plugin, Ratio
from qiskit_ibm_runtime.transpiler.passes.basis.to_nearest_clifford import ToNearestClifford
from qiskit_ibm_runtime.utils import validate_estimator_pubs, validate_isa_circuits


class Debugger:
    r"""A debugger.

    The debugger is a tool that a user of the Estimator primitive can use to understand the
    expected performance of their query.

    Given a backend and list of PUBs, the debugger takes the following steps:

    1. First, it simulates the estimation experiment in an ideal, noiseless setting, obtaining
      the ideal expectation values.
    2. Next, it simulates the estimation experiments in a noisy setting, obtaining a the noisy
      expectation values.
    3. Finally, it computes a figure of merit that captures the expected performance of an
      error-mitigation experiment performed on the chosen backend, for example the ration 
      ``noisy_vals/id_vals`` between the noisy and ideal expectation values (ratio).
    
    To ensure scalability, the simulations in steps 1 and 2 involve Clifford circuits obtained by
    applying the
    :meth:`~.qiskit_ibm_runtime.transpiler.passes.basis.to_nearest_clifford.ToNearestClifford`
    transpiler pass to the circuits in the PUBs. Additionally, the noise in step 2 is depolarizing
    noise built with ``NoiseModel.from_backend(backend, thermal_relaxation=False)``.
    
    Args:
        backend: A backend.
        plugin: A plugin that specifies the figure of merit returned by this debugger.
    """

    def __init__(self, backend: Union[BackendV1, BackendV2], plugin: Plugin = Ratio()) -> None:
        self._backend = backend
        self._plugin = plugin

    @property
    def backend(self):
        r"""
        The backend in this debugger.
        """
        return self._backend
    
    @property
    def plugin(self):
        r"""
        The plugin in this debugger.
        """
        return self._plugin

    def run(self, pubs: Iterable[EstimatorPubLike]):
        r"""
        Run the noisy and ideal simulation and compute the figure of merit.
        """
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]

        # validation
        validate_estimator_pubs(coerced_pubs)
        for pub in coerced_pubs:
            validate_isa_circuits([pub.circuit], self.backend.target)

        # cliffordization
        clifford_coerced_pubs = [
            EstimatorPub(
                PassManager([ToNearestClifford()]).run(pub.circuit),
                pub.observables,
                pub.parameter_values,
                pub.precision,
            )
            for pub in coerced_pubs
        ]

        # ideal simulation
        options = {"method": "stabilizer"}
        ideal_estimator = AerEstimator(options={"backend_options": options})
        ideal_results = ideal_estimator.run(clifford_coerced_pubs).result()

        # noisy simulation
        noise_model = NoiseModel.from_backend(self.backend, thermal_relaxation=False)
        options.update({"noise_model": noise_model})
        noisy_estimator = AerEstimator(options={"backend_options": options})
        noisy_results = noisy_estimator.run(clifford_coerced_pubs).result()

        return self.plugin(noisy_results, ideal_results)

    def __repr__(self) -> str:
        return f"Debugger(backend=\"{self.backend.name}\", plugin=\"{self.plugin.name}\")"
