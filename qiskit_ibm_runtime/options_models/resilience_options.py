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

"""Resilience options."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from qiskit.quantum_info import PauliLindbladMap

from .measure_noise_learning_options import MeasureNoiseLearningOptions
from .pec_options import PecOptions
from .utils import OptionsModel


class ResilienceOptions(OptionsModel):
    """Resilience options for V2 Estimator."""

    measure_mitigation: bool | None = None
    """Whether to enable measurement error mitigation method.

    If you enable measurement mitigation, you can fine-tune its noise learning
    by using :attr:`~measure_noise_learning`. See :class:`.~MeasureNoiseLearningOptions`
    for all measurement mitigation noise learning options.

    If ``measure_mitigation`` is ``None``, it is determined by the according to the resilience
    level: it is ``False`` for resilience level 0, and ``True`` for resilience levels 1 and 2.
    """

    measure_noise_learning: MeasureNoiseLearningOptions = Field(
        default_factory=MeasureNoiseLearningOptions
    )
    """Additional measurement noise learning options.

    See :class:`~.MeasureNoiseLearningOptions` for all options.
    """

    pec_mitigation: bool = False
    """Whether to turn on Probabilistic Error Cancellation error mitigation method.

    If you enable PEC, you can fine-tune its options by using :attr:`~pec`.
    See :class:`PecOptions` for additional PEC-related options.

    You must also provide a noise model via :attr:`~noise_model_mapping` when enabling PEC.
    """

    pec: PecOptions = Field(default_factory=PecOptions)
    """Additional probabalistic error cancellation mitigation options.

    See :class:`PecOptions` for all options.
    """

    zne_mitigation: bool | None = None
    """Whether to turn on Zero-Noise Extrapolation error mitigation method.

    If you enable ZNE, you can fine-tune its options by using :attr:`~zne`. See
    :class:`~.ZneOptions` for additional ZNE related options.

    If ``zne_mitigation`` is ``None``, it is determined by the server according to the resilience
    level: it is ``False`` for resilience levels ``0`` and ``1``, and ``True`` for resilience level
    ``2``.
    """

    noise_model_mapping: dict[str, PauliLindbladMap] | None = None
    """A noise model mapping for PEC mitigation.

    Maps layer references (strings) to :class:`~qiskit.quantum_info.PauliLindbladMap`
    objects that describe the noise characteristics of that layer. The dict contains
    layers from all PUBs. This is required when using PEC mitigation.
    """

    resilience_level: Literal[0, 1, 2] = 1
    """How much resilience to build against errors.

    Higher levels generate more accurate results, at the expense of longer processing times.

    * 0: No mitigation.
    * 1: Minimal mitigation costs. Mitigate error associated with readout errors.
    * 2: Medium mitigation costs. Typically reduces bias in estimators but it is not guaranteed to
        be zero bias.

    Refer to the
    `Configure error mitigation for Qiskit Runtime
    <https://quantum.cloud.ibm.com/docs/guides/configure-error-mitigation>`_ guide
    for more information about the error mitigation methods used at each level.
    """

    @field_validator("noise_model_mapping", mode="plain")
    @classmethod
    def _validate_noise_model_mapping(
        cls, value: dict[str, PauliLindbladMap] | None
    ) -> dict[str, PauliLindbladMap] | None:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError("'noise_model_mapping' must be a dict or None.")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"'noise_model_mapping' keys must be strings, got {type(k).__name__!r}."
                )
            if not isinstance(v, PauliLindbladMap):
                raise ValueError(
                    f"'noise_model_mapping' values must be PauliLindbladMap instances, "
                    f"got {type(v).__name__!r}."
                )
        return value
