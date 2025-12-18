"""Risk-mitigation guards for alignment, binner correctness, and white balance."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from training.evaluation import psnr
from training.losses import CFAPattern, quad_bayer_forward


def _ensure_image_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("expected input with shape (B, C, H, W) or (C, H, W)")
    return x


def detect_misalignment(
    anchor_mosaic: torch.Tensor,
    simulated_mosaic: torch.Tensor,
    *,
    max_shift: int = 2,
    psnr_threshold: float = 35.0,
) -> Dict[str, object]:
    """Search small shifts to detect obvious misalignment against the anchor.

    Returns a diagnostic dictionary including the best shift, the PSNR at that
    shift, the unshifted PSNR, and a boolean flag indicating whether the
    unshifted pair is within tolerance.
    """

    anchor = _ensure_image_4d(anchor_mosaic)
    simulated = _ensure_image_4d(simulated_mosaic)
    if anchor.shape != simulated.shape:
        raise ValueError("anchor_mosaic and simulated_mosaic must share shape")

    best_psnr = float("-inf")
    best_shift: Tuple[int, int] = (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = torch.roll(simulated, shifts=(dy, dx), dims=(-2, -1))
            score = float(psnr(shifted, anchor))
            if score > best_psnr:
                best_psnr = score
                best_shift = (dy, dx)

    reference_psnr = float(psnr(simulated, anchor))
    is_aligned = best_shift == (0, 0) and reference_psnr >= psnr_threshold

    return {
        "best_shift": best_shift,
        "best_psnr": best_psnr,
        "reference_psnr": reference_psnr,
        "is_aligned": is_aligned,
    }


def quad_binner_residual(
    rgb48: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    channel_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the max residual between the forward operator and manual binning."""

    if rgb48.ndim != 4 or rgb48.shape[1] != 3:
        raise ValueError("rgb48 must have shape (B, 3, H, W)")

    mosaiced = quad_bayer_forward(rgb48, cfa_pattern=cfa_pattern, channel_scale=channel_scale)

    if channel_scale is None:
        channel_scale = torch.ones(3, device=rgb48.device, dtype=rgb48.dtype)
    if channel_scale.shape != (3,):
        raise ValueError("channel_scale must have shape (3,)")

    scaled = rgb48 * channel_scale.view(1, 3, 1, 1)
    r = scaled[:, 0, 0::2, 0::2]
    g1 = scaled[:, 1, 0::2, 1::2]
    g2 = scaled[:, 1, 1::2, 0::2]
    b = scaled[:, 2, 1::2, 1::2]
    manual = torch.stack([r, g1, g2, b], dim=0).mean(dim=0).unsqueeze(1)

    return (mosaiced - manual).abs().max()


def _bayer_channel_map(height: int, width: int, pattern: CFAPattern) -> torch.Tensor:
    y_coords = torch.arange(height)
    x_coords = torch.arange(width)
    yy = y_coords.unsqueeze(1).expand(height, width) & 1
    xx = x_coords.unsqueeze(0).expand(height, width) & 1

    if pattern == "rggb":
        channel_map = torch.where((yy == 0) & (xx == 0), 0, torch.tensor(1))
        channel_map = torch.where((yy == 1) & (xx == 1), 2, channel_map)
    elif pattern == "bggr":
        channel_map = torch.where((yy == 0) & (xx == 0), 2, torch.tensor(1))
        channel_map = torch.where((yy == 1) & (xx == 1), 0, channel_map)
    elif pattern == "grbg":
        channel_map = torch.where((yy == 0) & (xx == 1), 0, torch.tensor(1))
        channel_map = torch.where((yy == 1) & (xx == 0), 2, channel_map)
    else:  # gbrg
        channel_map = torch.where((yy == 0) & (xx == 1), 2, torch.tensor(1))
        channel_map = torch.where((yy == 1) & (xx == 0), 0, channel_map)

    return channel_map.long()


def estimate_white_balance_neutral(
    mosaic: torch.Tensor, *, cfa_pattern: CFAPattern = "rggb", eps: float = 1e-6
) -> torch.Tensor:
    """Estimate AsShotNeutral-style ratios from a mosaiced RAW frame."""

    mosaic = _ensure_image_4d(mosaic)
    if mosaic.shape[1] != 1:
        raise ValueError("mosaic must have shape (B, 1, H, W)")

    _, _, height, width = mosaic.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("mosaic height and width must be even")

    channel_map = _bayer_channel_map(height, width, cfa_pattern).to(mosaic.device)
    values = mosaic[:, 0]
    mask = channel_map.unsqueeze(0).expand(values.shape)

    channel_means = []
    for idx in range(3):
        channel_vals = torch.where(mask == idx, values, torch.tensor(0.0, device=values.device, dtype=values.dtype))
        counts = (mask == idx).sum(dim=(1, 2)).clamp(min=1)
        channel_means.append(channel_vals.sum(dim=(1, 2)) / counts)

    means = torch.stack(channel_means, dim=1)
    neutral = 1.0 / (means + eps)
    neutral = neutral / neutral[:, 1:2]
    return neutral


def white_balance_consistency(
    mosaic: torch.Tensor,
    target_neutral: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    tolerance: float = 0.05,
) -> Tuple[bool, Dict[str, torch.Tensor]]:
    """Check that estimated neutral ratios stay near a target neutral vector."""

    estimated = estimate_white_balance_neutral(mosaic, cfa_pattern=cfa_pattern)
    target = target_neutral.view(1, 3).to(estimated.device, estimated.dtype)
    target = target / target[:, 1:2]

    deviation = (estimated - target).abs()
    within_tolerance = bool(torch.all(deviation <= tolerance))

    return within_tolerance, {"estimated": estimated, "deviation": deviation}

