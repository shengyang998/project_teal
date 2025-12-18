import torch

from training import analysis
from training.losses import quad_bayer_forward


def test_build_analysis_artifacts_projects_with_channel_scale():
    anchor_mosaic = torch.zeros((1, 1, 2, 2))
    candidate_rgb = torch.ones((1, 3, 4, 4))
    baseline_rgb = torch.zeros((1, 3, 4, 4))
    channel_scale = torch.tensor([1.0, 2.0, 4.0])

    expected_candidate = quad_bayer_forward(
        candidate_rgb, channel_scale=channel_scale
    )

    artifacts = analysis.build_analysis_artifacts(
        anchor_mosaic=anchor_mosaic,
        candidate_rgb48=candidate_rgb,
        baseline_rgb48=baseline_rgb,
        channel_scale=channel_scale,
    )

    assert torch.allclose(artifacts.mosaics["candidate"], expected_candidate)
    assert torch.allclose(artifacts.residuals["candidate"], expected_candidate - anchor_mosaic)
    assert set(artifacts.histograms.keys()) == {"candidate", "baseline"}


def test_build_analysis_artifacts_emits_renders_and_gain_map():
    anchor_mosaic = torch.full((1, 1, 2, 2), 0.5)
    candidate_rgb = torch.ones((1, 3, 4, 4))
    baseline_rgb = torch.zeros((1, 3, 4, 4))
    proraw_rgb = torch.full((1, 3, 4, 4), 0.25)
    anchor_render = torch.full((1, 3, 4, 4), 0.75)
    gain_map = torch.linspace(0.5, 1.5, steps=16).view(1, 1, 4, 4)

    def tone_curve(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    ccm = torch.eye(3) * 2.0

    artifacts = analysis.build_analysis_artifacts(
        anchor_mosaic=anchor_mosaic,
        candidate_rgb48=candidate_rgb,
        baseline_rgb48=baseline_rgb,
        proraw_rgb48=proraw_rgb,
        anchor_render=anchor_render,
        gain_map=gain_map,
        ccm=ccm,
        tone_curve=tone_curve,
    )

    expected_candidate_render = tone_curve(torch.einsum("ij,bcxy->bcxy", ccm, candidate_rgb))
    assert torch.allclose(artifacts.renders["candidate"], expected_candidate_render)
    assert torch.allclose(
        artifacts.renders["input"], tone_curve(torch.einsum("ij,bcxy->bcxy", ccm, proraw_rgb))
    )
    assert "gain" in artifacts.renders
    assert artifacts.renders["gain"].min() >= 0.0 and artifacts.renders["gain"].max() <= 1.0
