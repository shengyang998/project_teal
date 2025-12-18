"""Baseline paths for risk mitigation.

Baseline A is a global de-LTM fallback that rescales ProRAW48 into the RAW
anchor exposure using a robust percentile gain. It assumes a monotonic tone
curve can be inverted globally and avoids any spatial variation so the path is
fast and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.losses import CFAPattern, quad_bayer_forward


@dataclass(frozen=True)
class BaselineOutput:
    """Outputs for a baseline mapping.

    Attributes:
        rgb48: Linear 48MP RGB after baseline correction.
        mosaic: Forward-operator projection into RAW anchor space.
        gain: Scalar gain applied per batch element.
    """

    rgb48: torch.Tensor
    mosaic: torch.Tensor
    gain: torch.Tensor


def _validate_shapes(proraw_rgb48: torch.Tensor, anchor_mosaic: torch.Tensor) -> None:
    if proraw_rgb48.ndim != 4 or proraw_rgb48.shape[1] != 3:
        raise ValueError("proraw_rgb48 must have shape (B, 3, H, W)")
    if anchor_mosaic.ndim != 4 or anchor_mosaic.shape[1] != 1:
        raise ValueError("anchor_mosaic must have shape (B, 1, H/2, W/2)")

    batch, _, height, width = proraw_rgb48.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("proraw_rgb48 height and width must be even for binning")

    if anchor_mosaic.shape[0] != batch:
        raise ValueError("batch size mismatch between proraw_rgb48 and anchor_mosaic")


def global_del_tm_baseline(
    proraw_rgb48: torch.Tensor,
    anchor_mosaic: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    inverse_tone_curve=None,
    channel_scale: torch.Tensor | None = None,
    gain_percentile: float = 0.5,
    min_gain: float = 0.25,
    max_gain: float = 4.0,
    eps: float = 1e-6,
) -> BaselineOutput:
    """Compute Baseline A: global de-LTM gain alignment.

    The baseline:
    1. Optionally inverts a global tone curve on the ProRAW48 input.
    2. Projects the linearized frame into RAW anchor space via the forward
       operator.
    3. Estimates a robust scalar gain (per batch) to align exposures against the
       anchor mosaic.
    4. Applies the gain to the linear frame and re-projects into RAW space.
    """

    _validate_shapes(proraw_rgb48, anchor_mosaic)

    linear_rgb = proraw_rgb48 if inverse_tone_curve is None else inverse_tone_curve(proraw_rgb48)

    simulated_anchor = quad_bayer_forward(
        linear_rgb, cfa_pattern=cfa_pattern, channel_scale=channel_scale
    )
    if simulated_anchor.shape != anchor_mosaic.shape:
        raise ValueError("anchor_mosaic must match forward-projected shape")

    ratios = anchor_mosaic / (simulated_anchor + eps)
    gain = torch.quantile(ratios.view(ratios.shape[0], -1), gain_percentile, dim=1)
    gain = gain.clamp(min=min_gain, max=max_gain)

    gain_reshaped = gain.view(-1, 1, 1, 1)
    corrected_rgb = torch.clamp(linear_rgb * gain_reshaped, min=0.0)
    baseline_mosaic = quad_bayer_forward(
        corrected_rgb, cfa_pattern=cfa_pattern, channel_scale=channel_scale
    )

    return BaselineOutput(rgb48=corrected_rgb, mosaic=baseline_mosaic, gain=gain)

