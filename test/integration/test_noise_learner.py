# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Integration tests for NoiseLearner."""

from qiskit.circuit import QuantumCircuit
from qiskit_ibm_runtime import NoiseLearner

from ..decorators import run_integration_test
from ..ibm_test_case import IBMIntegrationTestCase


class TestIntegrationNoiseLearner(IBMIntegrationTestCase):
    """Integration tests for NoiseLearner."""

    def setUp(self) -> None:
        super().setUp()
        self.backend = "ibmq_qasm_simulator"

        c1 = QuantumCircuit(2)
        c1.cx(0, 1)

        c2 = QuantumCircuit(3)
        c2.cx(0, 1)
        c2.cx(1, 2)

        self.circuits = [c1, c2]

    @run_integration_test
    def test_noise_learner_session(self):
        """Verify if estimator primitive returns expected results"""

        learner = NoiseLearner(self.backend)
        learner.run(self.circuits)