"""Reporting helpers to compare candidate outputs against a baseline.

This module consumes the metric primitives in ``training.evaluation`` and
produces structured summaries that can be serialized or logged. The goal is to
make it easy to track whether the learned model beats a global inverse tone
curve baseline on anchor-space fidelity and edge consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

from training import evaluation
from training.qualitative import QualitativeSet

MetricDict = Dict[str, float]


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


@dataclass(frozen=True)
class SampleComparison:
    """Container for candidate vs. baseline metrics for a capture."""

    sample_id: str
    candidate: MetricDict
    baseline: MetricDict

    def delta(self) -> MetricDict:
        keys = set(self.candidate.keys()) & set(self.baseline.keys())
        return {key: self.candidate[key] - self.baseline[key] for key in keys}


def evaluate_sample(
    *,
    sample_id: str,
    candidate_mosaic: torch.Tensor,
    baseline_mosaic: torch.Tensor,
    anchor_mosaic: torch.Tensor,
    candidate_render: torch.Tensor,
    baseline_render: torch.Tensor,
    anchor_render: torch.Tensor,
    cfa_pattern: evaluation.CFAPattern = "rggb",
    ccm: torch.Tensor | None = None,
    tone_curve: callable | None = None,
) -> SampleComparison:
    """Compute anchor and edge metrics for a capture.

    Args:
        sample_id: Identifier for the capture.
        candidate_mosaic: Model output mosaiced into the RAW space.
        baseline_mosaic: Baseline (e.g., global inverse tone curve) mosaiced output.
        anchor_mosaic: Ground truth RAW anchor mosaic.
        candidate_render: Candidate linear RGB render for edge checks.
        baseline_render: Baseline render for edge checks.
        anchor_render: Reference render for edge checks.
        cfa_pattern: CFA pattern used for per-channel histograms/metrics.
        ccm: Optional color correction matrix applied before edge metrics.
        tone_curve: Optional tone curve applied before edge metrics.

    Returns:
        SampleComparison with scalar metrics for candidate and baseline paths.
    """

    candidate_metrics: MetricDict = {
        "psnr": _to_float(evaluation.psnr(candidate_mosaic, anchor_mosaic)),
        "ssim": _to_float(evaluation.ssim(candidate_mosaic, anchor_mosaic)),
    }
    candidate_grad = evaluation.gradient_consistency_metrics(
        candidate_render, anchor_render, ccm=ccm, tone_curve=tone_curve
    )
    candidate_metrics.update({k: _to_float(v) for k, v in candidate_grad.items()})

    baseline_metrics: MetricDict = {
        "psnr": _to_float(evaluation.psnr(baseline_mosaic, anchor_mosaic)),
        "ssim": _to_float(evaluation.ssim(baseline_mosaic, anchor_mosaic)),
    }
    baseline_grad = evaluation.gradient_consistency_metrics(
        baseline_render, anchor_render, ccm=ccm, tone_curve=tone_curve
    )
    baseline_metrics.update({k: _to_float(v) for k, v in baseline_grad.items()})

    return SampleComparison(sample_id=sample_id, candidate=candidate_metrics, baseline=baseline_metrics)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot compute mean of empty sequence")
    return sum(vals) / len(vals)


def summarize_samples(samples: Sequence[SampleComparison]) -> Dict[str, Dict[str, float]]:
    """Aggregate candidate/baseline metrics across samples."""

    if not samples:
        raise ValueError("no samples provided for summary")

    metric_names = set().union(*(s.candidate.keys() for s in samples))
    summary: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        candidate_vals = [_to_float(s.candidate[name]) for s in samples if name in s.candidate]
        baseline_vals = [_to_float(s.baseline[name]) for s in samples if name in s.baseline]
        if not candidate_vals or not baseline_vals:
            continue
        summary[name] = {
            "candidate": _mean(candidate_vals),
            "baseline": _mean(baseline_vals),
            "delta": _mean(cv - bv for cv, bv in zip(candidate_vals, baseline_vals)),
        }
    return summary


def generate_report(
    samples: Sequence[SampleComparison],
    *,
    qualitative_set: QualitativeSet | None = None,
) -> Dict[str, object]:
    """Build a structured report comparing candidate and baseline metrics."""

    summary = summarize_samples(samples)
    sample_entries: List[Dict[str, object]] = []
    for sample in samples:
        sample_entries.append(
            {
                "id": sample.sample_id,
                "candidate": sample.candidate,
                "baseline": sample.baseline,
                "delta": sample.delta(),
            }
        )

    report: Dict[str, object] = {"summary": summary, "samples": sample_entries}

    if qualitative_set is not None:
        report["qualitative"] = {
            "num_samples": len(qualitative_set.samples),
            "tag_counts": qualitative_set.tag_counts(),
        }

    return report
