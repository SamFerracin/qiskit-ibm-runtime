# This code is part of Qiskit.
#
# (C) Copyright IBM 2025-2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Converters for executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..datatree import DataTree

from typing import Protocol, TypeVar

T = TypeVar("T")


class PassthroughDataAdapter(Protocol[T]):
    """A protocol defining version-specific conversion operations used for passthrough data."""

    def tensor_model_from_numpy(self, value: np.ndarray) -> T:
        """Convert from numpy to a schema model."""

    def tensor_model_to_numpy(self, value: T) -> np.ndarray:
        """Convert from a schema model to numpy."""


def passthrough_data_to_schema(passthrough_data: DataTree, adapter: PassthroughDataAdapter[T]):  # type: ignore[no-untyped-def]
    """Convert passthrough data to schema model."""
    if isinstance(passthrough_data, dict):
        return {k: passthrough_data_to_schema(v, adapter) for k, v in passthrough_data.items()}

    if isinstance(passthrough_data, (list, tuple)):
        return [passthrough_data_to_schema(v, adapter) for v in passthrough_data]

    if isinstance(passthrough_data, np.ndarray):
        return {
            "__type__": "array",
            "__value__": adapter.tensor_model_from_numpy(passthrough_data),
        }

    return passthrough_data


def passthrough_data_from_schema(
    passthrough_data: object, adapter: PassthroughDataAdapter[T]
) -> DataTree:
    """Convert passthrough data from schema model."""
    print("hello")
    if isinstance(passthrough_data, dict):
        try:
            type = passthrough_data["__type__"]
            print(type)
            if type == "array":
                print(passthrough_data["__value__"])
                return adapter.tensor_model_to_numpy(passthrough_data["__value__"])
            raise ValueError(f"Cannot deserialize object of type {type}.")
        except KeyError:
            return {
                k: passthrough_data_from_schema(v, adapter) for k, v in passthrough_data.items()
            }

    if isinstance(passthrough_data, list):
        return [passthrough_data_from_schema(v, adapter) for v in passthrough_data]
    return passthrough_data
