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

"""NoiseLearner result class"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import PauliList


class LindbladErrors:
    """A container for the generators and rates for a single-layer noise process
    in Pauli Lindblad format.

    Args:
        generators: The list of generators for the Pauli errors.
        rates: The rates for the given Pauli error generators.

    Raises:
        ValueError: If ``generators`` and ``rates`` have different lengths.
    """

    def __init__(self, generators: PauliList, rates: Sequence[float]) -> None:
        if len(generators) != len(rates):
            msg = f"``generators`` has length {len(generators)}, but "
            msg += f"``rates`` has length {len(rates)}."
            raise ValueError(msg)

        self._generators = generators
        self._rates = rates

    @property
    def generators(self):
        r"""
        The generators of this :class:`LindbladErrors`.
        """
        return self._generators

    @property
    def rates(self):
        r"""
        The rates of this :class:`LindbladErrors`.
        """
        return self._rates

    @property
    def num_qubits(self):
        r"""
        The number of qubits in this :class:`LindbladErrors`.
        """
        return self.generators.num_qubits
    
    def __repr__(self) -> str:
        return f"LindbladErrors(generators={self.generators}, rates={self.rates})"


class LayerNoise:
    """The noise (in Pauli Lindblad format) of a single layer of instructions.

    Args:
        circuit: A circuit whose noise has been learnt.
        qubits: The labels of the qubits in the ``circuit``.
        errors: The Pauli Lindblad errors affecting the ``circuit``.

    Raises:
        ValueError: If ``circuit``, ``qubits``, and ``errors`` have mismatching number of
            qubits.
    """

    def __init__(
        self, circuit: QuantumCircuit, qubits: Sequence[int], errors: LindbladErrors
    ) -> None:
        if len({circuit.num_qubits, len(qubits), errors.num_qubits}) != 1:
            raise ValueError("Mistmatching numbers of qubits.")

        self._circuit = circuit
        self._qubits = qubits
        self._errors = errors

    @property
    def circuit(self):
        r"""
        The circuit in this :class:`LayerNoise`.
        """
        return self._circuit

    @property
    def qubits(self):
        r"""
        The qubits in this :class:`LayerNoise`.
        """
        return self._qubits

    @property
    def errors(self):
        r"""
        The errors in this :class:`LayerNoise`.
        """
        return self._errors
    
    def __repr__(self) -> str:
        ret = f"circuit={repr(self.circuit)}, qubits={self.qubits}, errors={self.errors})"
        return f"LayerNoise({ret})"


class NoiseLearnerResult:
    """A container for the results of a noise learner experiment."""

    def __init__(self, data: Sequence[LayerNoise], metadata: dict[str, Any] | None = None):
        """
        Args:
            data: The data of a noise learner experiment.
            metadata: Metadata that is common to all pub results; metadata specific to particular
                pubs should be placed in their metadata fields. Keys are expected to be strings.
        """
        self._data = list(data)
        self._metadata = metadata.copy() or {}

    @property
    def data(self) -> Sequence[LayerNoise]:
        """The data of this noise learner result."""
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of this noise learner result."""
        return self._metadata

    def __getitem__(self, index: int) -> LayerNoise:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"NoiseLearnerResult(data={self.data}, metadata={self.metadata})"

    def __iter__(self) -> Iterator[LayerNoise]:
        return iter(self.data)
