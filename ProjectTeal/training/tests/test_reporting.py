import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

from training import reporting
from training.qualitative import QualitativeSample, QualitativeSet


def test_evaluate_sample_computes_candidate_and_baseline_metrics():
    anchor_mosaic = torch.zeros(1, 1, 4, 4)
    candidate_mosaic = anchor_mosaic + 0.01
    baseline_mosaic = anchor_mosaic + 0.1

    anchor_render = torch.zeros(1, 3, 4, 4)
    candidate_render = anchor_render.clone()
    candidate_render[:, :, :, 2:] = 0.02
    baseline_render = anchor_render + 0.1

    comparison = reporting.evaluate_sample(
        sample_id="scene_a",
        candidate_mosaic=candidate_mosaic,
        baseline_mosaic=baseline_mosaic,
        anchor_mosaic=anchor_mosaic,
        candidate_render=candidate_render,
        baseline_render=baseline_render,
        anchor_render=anchor_render,
    )

    assert comparison.sample_id == "scene_a"
    assert comparison.candidate["psnr"] > comparison.baseline["psnr"]
    assert comparison.candidate["ssim"] > comparison.baseline["ssim"]
    assert comparison.candidate["halo_l1"] < comparison.baseline["halo_l1"]


def test_generate_report_summarizes_deltas_and_tags():
    anchor = torch.zeros(1, 1, 2, 2)

    sample_one = reporting.evaluate_sample(
        sample_id="scene_one",
        candidate_mosaic=anchor + 0.05,
        baseline_mosaic=anchor + 0.1,
        anchor_mosaic=anchor,
        candidate_render=torch.zeros(1, 3, 2, 2),
        baseline_render=torch.ones(1, 3, 2, 2) * 0.5,
        anchor_render=torch.zeros(1, 3, 2, 2),
    )
    sample_two = reporting.evaluate_sample(
        sample_id="scene_two",
        candidate_mosaic=anchor + 0.02,
        baseline_mosaic=anchor + 0.05,
        anchor_mosaic=anchor,
        candidate_render=torch.zeros(1, 3, 2, 2),
        baseline_render=torch.ones(1, 3, 2, 2) * 0.25,
        anchor_render=torch.zeros(1, 3, 2, 2),
    )

    qualitative = QualitativeSet(
        [
            QualitativeSample("scene_one", path=Path("/tmp/a"), tags=["backlit"]),
            QualitativeSample("scene_two", path=Path("/tmp/b"), tags=["foliage"]),
        ]
    )

    report = reporting.generate_report([sample_one, sample_two], qualitative_set=qualitative)

    assert set(report.keys()) == {"summary", "samples", "qualitative"}
    assert report["summary"]["psnr"]["candidate"] > report["summary"]["psnr"]["baseline"]
    assert report["summary"]["ssim"]["delta"] > 0
    assert report["qualitative"]["num_samples"] == 2
    assert report["qualitative"]["tag_counts"]["backlit"] == 1
