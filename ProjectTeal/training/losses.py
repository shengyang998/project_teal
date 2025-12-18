"""Losses and differentiable operators for the 48MP linear RGB pipeline."""

from __future__ import annotations

from typing import Callable, Dict, Literal, Tuple

import torch
import torch.nn.functional as F

CFAPattern = Literal["rggb", "bggr", "grbg", "gbrg"]


def _channel_index_map(height: int, width: int, pattern: CFAPattern) -> torch.Tensor:
    """Create a channel index map for quad-Bayer mosaicing.

    Args:
        height: Height of the high-resolution image.
        width: Width of the high-resolution image.
        pattern: CFA pattern for the quad-Bayer blocks.

    Returns:
        Tensor of shape (height, width) with values in {0, 1, 2}.
    """

    y_coords = torch.arange(height)
    x_coords = torch.arange(width)
    block_y = ((y_coords // 2) & 1).unsqueeze(1)
    block_x = ((x_coords // 2) & 1).unsqueeze(0)

    # Expand to full grid for logical operations.
    bx = block_x.expand(height, width)
    by = block_y.expand(height, width)

    if pattern == "rggb":
        channel_map = torch.where((bx == 0) & (by == 0), 0, torch.tensor(2))
        channel_map = torch.where((bx == 1) & (by == 0), 1, channel_map)
        channel_map = torch.where((bx == 0) & (by == 1), 1, channel_map)
    elif pattern == "bggr":
        channel_map = torch.where((bx == 0) & (by == 0), 2, torch.tensor(1))
        channel_map = torch.where((bx == 1) & (by == 0), 1, channel_map)
        channel_map = torch.where((bx == 0) & (by == 1), 1, channel_map)
    elif pattern == "grbg":
        channel_map = torch.where((bx == 0) & (by == 0), 1, torch.tensor(1))
        channel_map = torch.where((bx == 1) & (by == 0), 0, channel_map)
        channel_map = torch.where((bx == 0) & (by == 1), 2, channel_map)
    else:  # gbrg
        channel_map = torch.where((bx == 0) & (by == 0), 1, torch.tensor(1))
        channel_map = torch.where((bx == 1) & (by == 0), 2, channel_map)
        channel_map = torch.where((bx == 0) & (by == 1), 0, channel_map)

    return channel_map.long()


def quad_bayer_forward(
    rgb48: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    channel_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mosaic and bin a 48MP RGB prediction to the 12MP RAW anchor domain.

    Args:
        rgb48: Tensor of shape (B, 3, H, W) representing linear 48MP RGB.
        cfa_pattern: Quad-Bayer CFA pattern; defaults to RGGB.
        channel_scale: Optional per-channel scaling tensor of shape (3,).

    Returns:
        Tensor of shape (B, 1, H/2, W/2) representing the mosaiced RAW anchor.
    """

    if rgb48.ndim != 4 or rgb48.shape[1] != 3:
        raise ValueError("rgb48 must have shape (B, 3, H, W)")

    batch, _, height, width = rgb48.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Input dimensions must be even for 2x2 binning")

    if channel_scale is None:
        channel_scale = torch.ones(3, device=rgb48.device, dtype=rgb48.dtype)
    if channel_scale.shape != (3,):
        raise ValueError("channel_scale must have shape (3,)")

    channel_map = _channel_index_map(height, width, cfa_pattern).to(rgb48.device)
    scaled = rgb48 * channel_scale.view(1, 3, 1, 1)

    gathered = scaled.gather(1, channel_map.expand(batch, -1, -1).unsqueeze(1))
    mosaiced = gathered.view(batch, 1, height // 2, 2, width // 2, 2).mean(dim=(3, 5))
    return mosaiced


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Compute the Charbonnier loss between prediction and target."""

    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def anchor_charbonnier_loss(
    pred_rgb48: torch.Tensor,
    target_mosaic: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    channel_scale: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Anchor loss comparing forward-operator output to RAW mosaic."""

    predicted_mosaic = quad_bayer_forward(
        pred_rgb48, cfa_pattern=cfa_pattern, channel_scale=channel_scale
    )

    if target_mosaic.shape != predicted_mosaic.shape:
        raise ValueError("target_mosaic must match forward operator output shape")

    return charbonnier_loss(predicted_mosaic, target_mosaic, eps=eps)


def _apply_global_render(
    rgb: torch.Tensor,
    *,
    ccm: torch.Tensor | None = None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    if ccm is None:
        ccm = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
    if ccm.shape != (3, 3):
        raise ValueError("ccm must have shape (3, 3)")

    rendered = torch.einsum("ij,bcxy->bcxy", ccm, rgb)
    if tone_curve is not None:
        rendered = tone_curve(rendered)
    return rendered


def _spatial_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


def gradient_detail_loss(
    pred_rgb48: torch.Tensor,
    reference_rgb48: torch.Tensor,
    *,
    ccm: torch.Tensor | None = None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Gradient loss after global-only rendering (tone curve + CCM)."""

    if pred_rgb48.shape != reference_rgb48.shape:
        raise ValueError("pred_rgb48 and reference_rgb48 must share shape")

    pred_render = _apply_global_render(pred_rgb48, ccm=ccm, tone_curve=tone_curve)
    ref_render = _apply_global_render(reference_rgb48, ccm=ccm, tone_curve=tone_curve)

    pred_dx, pred_dy = _spatial_gradients(pred_render)
    ref_dx, ref_dy = _spatial_gradients(ref_render)

    return (pred_dx - ref_dx).abs().mean() + (pred_dy - ref_dy).abs().mean()


def gain_regularization(
    gain_map: torch.Tensor,
    *,
    smoothness_weight: float = 1.0,
    min_gain: float = 0.5,
    max_gain: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Smoothness + range regularization for gain fields."""

    if gain_map.ndim != 4:
        raise ValueError("gain_map must have shape (B, C, H, W)")

    dx, dy = _spatial_gradients(gain_map)
    smoothness = dx.abs().mean() + dy.abs().mean()

    lower_penalty = torch.clamp_min(min_gain - gain_map, 0)
    upper_penalty = torch.clamp_min(gain_map - max_gain, 0)
    range_penalty = (lower_penalty + upper_penalty).mean()

    total = smoothness_weight * smoothness + range_penalty
    components = {"smoothness": smoothness, "range": range_penalty}
    return total, components
