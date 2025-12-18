"""Analysis utilities for CI parity with RAW ingest and gain-field paths.

These helpers keep Python/CI visualizations (histograms, side-by-sides)
consistent with the sensor-consistency forward operator and the gain-field
path used during training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch

from training import evaluation
from training.losses import CFAPattern, quad_bayer_forward


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Container for rendered analysis outputs."""

    mosaics: Dict[str, torch.Tensor]
    residuals: Dict[str, torch.Tensor]
    histograms: Dict[str, Dict[str, torch.Tensor]]
    renders: Dict[str, torch.Tensor]


def _apply_render(
    rgb: torch.Tensor,
    *,
    ccm: torch.Tensor | None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    if ccm is None:
        ccm = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
    if ccm.shape != (3, 3):
        raise ValueError("ccm must have shape (3, 3)")

    rendered = torch.einsum("ij,bcxy->bcxy", ccm, rgb)
    if tone_curve is not None:
        rendered = tone_curve(rendered)
    return torch.clamp(rendered, min=0.0)


def _normalize_gain_map(gain_map: torch.Tensor) -> torch.Tensor:
    if gain_map.ndim != 4:
        raise ValueError("gain_map must have shape (B, C, H, W)")

    gain_display = gain_map
    if gain_display.shape[1] == 1:
        gain_display = gain_display.repeat(1, 3, 1, 1)

    flat = gain_display.view(gain_display.shape[0], gain_display.shape[1], -1)
    min_vals = flat.min(dim=2).values.view(gain_display.shape[0], gain_display.shape[1], 1, 1)
    max_vals = flat.max(dim=2).values.view(gain_display.shape[0], gain_display.shape[1], 1, 1)
    denom = (max_vals - min_vals).clamp_min(1e-6)
    return (gain_display - min_vals) / denom


def _render_stack(
    *,
    candidate_rgb48: torch.Tensor,
    baseline_rgb48: torch.Tensor,
    ccm: torch.Tensor | None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None,
    proraw_rgb48: torch.Tensor | None,
    anchor_render: torch.Tensor | None,
    gain_map: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    renders: Dict[str, torch.Tensor] = {
        "candidate": _apply_render(candidate_rgb48, ccm=ccm, tone_curve=tone_curve),
        "baseline": _apply_render(baseline_rgb48, ccm=ccm, tone_curve=tone_curve),
    }

    if proraw_rgb48 is not None:
        renders["input"] = _apply_render(proraw_rgb48, ccm=ccm, tone_curve=tone_curve)
    if anchor_render is not None:
        renders["anchor"] = _apply_render(anchor_render, ccm=ccm, tone_curve=tone_curve)
    if gain_map is not None:
        renders["gain"] = _normalize_gain_map(gain_map)

    return renders


def build_analysis_artifacts(
    *,
    anchor_mosaic: torch.Tensor,
    candidate_rgb48: torch.Tensor,
    baseline_rgb48: torch.Tensor,
    proraw_rgb48: torch.Tensor | None = None,
    anchor_render: torch.Tensor | None = None,
    gain_map: torch.Tensor | None = None,
    cfa_pattern: CFAPattern = "rggb",
    channel_scale: torch.Tensor | None = None,
    ccm: torch.Tensor | None = None,
    tone_curve: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> AnalysisArtifacts:
    """Assemble mosaics, histograms, and side-by-sides for CI analysis."""

    candidate_mosaic = quad_bayer_forward(
        candidate_rgb48, cfa_pattern=cfa_pattern, channel_scale=channel_scale
    )
    baseline_mosaic = quad_bayer_forward(
        baseline_rgb48, cfa_pattern=cfa_pattern, channel_scale=channel_scale
    )

    if anchor_mosaic.shape != candidate_mosaic.shape:
        raise ValueError("anchor_mosaic must match forward-projected shape")

    residuals = {
        "candidate": candidate_mosaic - anchor_mosaic,
        "baseline": baseline_mosaic - anchor_mosaic,
    }

    histograms = {
        name: evaluation.per_cfa_error_histogram(residuals[name], cfa_pattern=cfa_pattern)
        for name in residuals
    }

    renders = _render_stack(
        candidate_rgb48=candidate_rgb48,
        baseline_rgb48=baseline_rgb48,
        ccm=ccm,
        tone_curve=tone_curve,
        proraw_rgb48=proraw_rgb48,
        anchor_render=anchor_render,
        gain_map=gain_map,
    )

    mosaics = {
        "candidate": candidate_mosaic,
        "baseline": baseline_mosaic,
        "anchor": anchor_mosaic,
    }

    return AnalysisArtifacts(
        mosaics=mosaics,
        residuals=residuals,
        histograms=histograms,
        renders=renders,
    )
