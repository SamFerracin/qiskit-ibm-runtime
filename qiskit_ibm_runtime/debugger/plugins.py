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
from abc import ABC, abstractmethod
import numpy as np
from typing import Any

from qiskit.primitives.containers import PrimitiveResult


class Plugin(ABC):
    r"""Base class for the debugger's plugins.

    Args:
        name: The name of this plugin.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self):
        r"""
        The name of this plugin.
        """
        return self._name

    @abstractmethod
    def call(self, noisy_results: PrimitiveResult, ideal_results: PrimitiveResult) -> Any:
        r"""
        The calculation performed by this plugin to compute the figure of merit.
        """
        raise NotImplementedError()

    def __call__(self, noisy_results: PrimitiveResult, ideal_results: PrimitiveResult) -> Any:
        return self.call(noisy_results, ideal_results)

    def __repr__(self) -> str:
        return f"{self.__class__.name}({self.name})"


class Ratio(Plugin):
    r"""A :meth:`.~Plugin` that computes the ratio ``noisy_results/ideal_results`` between the
    noisy expectation values in ``noisy_results`` and the ideal ones in ``ideal_results``.

    Returns ``0`` when it encounters an ideal value equal to ``0``.
    """

    def __init__(self) -> None:
        super().__init__(name="ratio")

    def call(self, noisy_results: PrimitiveResult, ideal_results: PrimitiveResult):
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
