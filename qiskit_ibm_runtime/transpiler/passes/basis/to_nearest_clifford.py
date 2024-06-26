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

"""Pass to convert Id gate operations to a delay instruction."""

from typing import Dict

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit.circuit import ControlFlowOp
from qiskit.circuit import Delay
from qiskit.circuit.library import CXGate, CZGate, ECRGate, RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.instruction_durations import InstructionDurations

SUPPORTED_GATES = (CXGate, CZGate, ECRGate, RZGate, SXGate, XGate)

class ToNearestClifford(TransformationPass):
    """
    Convert :class:`qiskit.circuit.gate.Gate`\s to the nearest Clifford gate.

    This pass can be used to uniquely map an ISA circuit to a Clifford circuit. To do so,
    it replaces every :class:`qiskit.circuit.library.RZGate` by angle :math:`\phi`
    with a corresponding rotation by angle :math:`\phi'`, where :math:`\phi'` is the
    multiple of :math:`\pi/2` closest to :math:`\phi`\. It skips
    :class:`qiskit.circuit.library.CXGate`\s, :class:`qiskit.circuit.library.CZGate`\s,
    :class:`qiskit.circuit.library.ECRGate`\s, :class:`qiskit.circuit.library.SXGate`\s,
    and :class:`qiskit.circuit.library.XGate`\s, and it errors for every other gate.
    """

    def __init__(self):
        """Convert :class:`qiskit.circuit.gate.Gate`\s to the nearest Clifford gate.
        """
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes():
            if not isinstance(node.op, SUPPORTED_GATES):
                msg = f"Found gate ``{node.op.__class__.__name__}``, but only gates"
                msg += f" {[g for g in SUPPORTED_GATES]} are supported."
                raise ValueError(msg)
        return dag