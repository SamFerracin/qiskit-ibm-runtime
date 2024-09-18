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

"""Tests for the ``Embedding`` class."""

import plotly.graph_objects as go

from qiskit_aer import AerSimulator

from qiskit_ibm_runtime.fake_provider.local_service import QiskitRuntimeLocalService
from qiskit_ibm_runtime.utils.embeddings import Embedding

from ..ibm_test_case import IBMTestCase


class TestEmbedding(IBMTestCase):
    """Class for testing the Embedding class."""

    def setUp(self):
        super().setUp()

        service = QiskitRuntimeLocalService()
        self.aer = AerSimulator()
        self.kyiv = service.backend("fake_kyiv")
        self.vigo = service.backend("fake_vigo")

    def test_from_backend(self):
        r"""Test the constructor from backend."""
        e = Embedding.from_backend(self.vigo)

        coo = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]
        self.assertEqual(e.coordinates, coo)
        self.assertEqual(e.coupling_map, self.vigo.coupling_map)

    def test_init_error(self):
        r"""Test the errors raised by the constructor."""
        e_vigo = Embedding.from_backend(self.vigo)
        e_kyiv = Embedding.from_backend(self.kyiv)

        with self.assertRaises(ValueError):
            Embedding.from_backend(self.aer)

        with self.assertRaises(ValueError):
            Embedding(e_vigo.coordinates, e_kyiv.coupling_map)
