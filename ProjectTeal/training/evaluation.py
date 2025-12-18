"""Evaluation metrics for the 48MP linear RGB pipeline.

The helpers here focus on anchor-space fidelity (PSNR/SSIM on the mosaiced
RAW) and edge consistency (gradient correlation/halo indicators after global
rendering). Per-CFA error histograms make it easier to spot color-channel
drift.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Tuple

import torch
import torch.nn.functional as F

CFAPattern = Literal["rggb", "bggr", "grbg", "gbrg"]


def _assert_matching_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape")


def _ensure_image_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("expected input with shape (B, C, H, W) or (C, H, W)")
    return x


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio for image tensors."""

    pred = _ensure_image_4d(pred)
    target = _ensure_image_4d(target)
    _assert_matching_shapes(pred, target)

    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device, dtype=pred.dtype)
    return 10 * torch.log10(max_val**2 / mse)


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    window_1d = gauss / gauss.sum()
    window_2d = torch.outer(window_1d, window_1d)
    return window_2d.unsqueeze(0).unsqueeze(0)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Compute Structural Similarity (SSIM) index for images."""

    pred = _ensure_image_4d(pred)
    target = _ensure_image_4d(target)
    _assert_matching_shapes(pred, target)

    if pred.shape[1] != target.shape[1]:
        raise ValueError("pred and target must have the same number of channels")

    padding = window_size // 2
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window.expand(pred.shape[1], 1, window_size, window_size)

    def _filter(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=padding, groups=x.shape[1])

    mu_pred = _filter(pred)
    mu_target = _filter(target)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = _filter(pred * pred) - mu_pred_sq
    sigma_target_sq = _filter(target * target) - mu_target_sq
    sigma_pred_target = _filter(pred * target) - mu_pred_target

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


def _bayer_channel_map(height: int, width: int, pattern: CFAPattern) -> torch.Tensor:
    y_coords = torch.arange(height)
    x_coords = torch.arange(width)
    yy = y_coords.unsqueeze(1).expand(height, width) & 1
    xx = x_coords.unsqueeze(0).expand(height, width) & 1

    if pattern == "rggb":
        # (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B
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


def per_cfa_error_histogram(
    residual_mosaic: torch.Tensor,
    *,
    cfa_pattern: CFAPattern = "rggb",
    num_bins: int = 64,
    value_range: Tuple[float, float] = (-0.02, 0.02),
) -> Dict[str, torch.Tensor]:
    """Compute per-channel error histograms for a mosaiced residual."""

    residual_mosaic = _ensure_image_4d(residual_mosaic)
    if residual_mosaic.shape[1] != 1:
        raise ValueError("residual_mosaic must have shape (B, 1, H, W)")

    _, _, height, width = residual_mosaic.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("mosaic height and width must be even")

    channel_map = _bayer_channel_map(height, width, cfa_pattern).to(residual_mosaic.device)
    residual = residual_mosaic.squeeze(1)

    histograms: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(["r", "g", "b"]):
        values = residual[channel_map == idx]
        if values.numel() == 0:
            hist = torch.zeros(num_bins, device=residual.device, dtype=residual.dtype)
        else:
            hist = torch.histc(values, bins=num_bins, min=value_range[0], max=value_range[1])
        histograms[name] = hist

    return histograms


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


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat) + eps
    return torch.dot(a_flat, b_flat) / denom


def gradient_consistency_metrics(
    pred_rgb: torch.Tensor,
    reference_rgb: torch.Tensor,
    *,
    ccm: torch.Tensor | None = None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Edge consistency metrics after applying a global render."""

    pred_rgb = _ensure_image_4d(pred_rgb)
    reference_rgb = _ensure_image_4d(reference_rgb)
    _assert_matching_shapes(pred_rgb, reference_rgb)

    pred_render = _apply_global_render(pred_rgb, ccm=ccm, tone_curve=tone_curve)
    ref_render = _apply_global_render(reference_rgb, ccm=ccm, tone_curve=tone_curve)

    pred_dx, pred_dy = _spatial_gradients(pred_render)
    ref_dx, ref_dy = _spatial_gradients(ref_render)

    grad_mag_pred = torch.sqrt(pred_dx.pow(2) + pred_dy.pow(2) + 1e-12)
    grad_mag_ref = torch.sqrt(ref_dx.pow(2) + ref_dy.pow(2) + 1e-12)

    corr_x = _cosine_similarity(pred_dx, ref_dx)
    corr_y = _cosine_similarity(pred_dy, ref_dy)
    halo_indicator = (grad_mag_pred - grad_mag_ref).abs().mean()

    return {
        "corr_x": corr_x,
        "corr_y": corr_y,
        "halo_l1": halo_indicator,
    }

