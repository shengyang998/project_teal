"""Input preparation and output composition helpers for gain-field model.

These utilities keep preprocessing logic close to the model so training and
export code share the same tensor layout assumptions.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def upsample_guidance(raw12_rgb: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Upsample RAW12 guidance RGB to match the 48MP spatial resolution.

    Args:
        raw12_rgb: Tensor of shape (B, 3, H, W) representing demosaiced RAW12
            guidance in linear space.
        target_size: (height, width) of the ProRAW48 input.

    Returns:
        Tensor of shape (B, 3, target_size[0], target_size[1]) using bilinear
        interpolation without aligning corners.
    """

    if raw12_rgb.ndim != 4 or raw12_rgb.shape[1] != 3:
        raise ValueError("raw12_rgb must have shape (B, 3, H, W)")

    return F.interpolate(raw12_rgb, size=target_size, mode="bilinear", align_corners=False)


def build_model_inputs(pro_raw48: torch.Tensor, raw12_rgb: torch.Tensor) -> torch.Tensor:
    """Concatenate ProRAW48 detail and RAW12 guidance into model input tensor.

    Args:
        pro_raw48: Tensor (B, 3, H, W) of linearized ProRAW48 RGB.
        raw12_rgb: Tensor (B, 3, h, w) of demosaiced RAW12 guidance.

    Returns:
        Tensor (B, 6, H, W) stacking ProRAW48 and upsampled RAW12 guidance.
    """

    if pro_raw48.ndim != 4 or pro_raw48.shape[1] != 3:
        raise ValueError("pro_raw48 must have shape (B, 3, H, W)")
    if raw12_rgb.ndim != 4 or raw12_rgb.shape[1] != 3:
        raise ValueError("raw12_rgb must have shape (B, 3, h, w)")
    if pro_raw48.shape[0] != raw12_rgb.shape[0]:
        raise ValueError("Batch size mismatch between pro_raw48 and raw12_rgb")

    h_ratio = pro_raw48.shape[2] / raw12_rgb.shape[2]
    w_ratio = pro_raw48.shape[3] / raw12_rgb.shape[3]
    if h_ratio != w_ratio or h_ratio < 1:
        raise ValueError("raw12 guidance must scale uniformly to ProRAW48 resolution")

    guidance_up = upsample_guidance(raw12_rgb, target_size=pro_raw48.shape[-2:])
    return torch.cat([pro_raw48, guidance_up], dim=1)


def compose_linear_prediction(
    pro_raw48: torch.Tensor,
    gain: torch.Tensor,
    detail: torch.Tensor,
    *,
    upsample_mode: str = "bilinear",
) -> torch.Tensor:
    """Apply gain field and residual to form the predicted 48MP linear RGB.

    Args:
        pro_raw48: Tensor (B, 3, H, W) of linearized ProRAW48 RGB input.
        gain: Tensor (B, 3, h, w) gain field (low-res inverse-LTM gains).
        detail: Tensor (B, 3, H, W) residual detail map at full resolution.
        upsample_mode: Interpolation mode used to upsample the gain field.

    Returns:
        Tensor (B, 3, H, W) representing \hat{L}_{48} = max(0, I_lin_48 ⊙ g + r).
    """

    if pro_raw48.ndim != 4 or detail.ndim != 4 or gain.ndim != 4:
        raise ValueError("All tensors must be 4D")
    if pro_raw48.shape != detail.shape:
        raise ValueError("pro_raw48 and detail must share shape (B, 3, H, W)")
    if pro_raw48.shape[1] != 3 or gain.shape[1] != 3:
        raise ValueError("pro_raw48, gain, and detail must have 3 channels")

    gain_up = F.interpolate(gain, size=pro_raw48.shape[-2:], mode=upsample_mode, align_corners=False)
    combined = pro_raw48 * gain_up + detail
    return torch.clamp_min(combined, 0.0)
