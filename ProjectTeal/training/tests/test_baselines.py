import torch

from training import baselines
from training.losses import quad_bayer_forward


def test_global_del_tm_aligns_anchor_with_scalar_gain():
    proraw = torch.full((1, 3, 4, 4), 0.5)
    anchor = torch.full((1, 1, 2, 2), 1.0)

    output = baselines.global_del_tm_baseline(proraw, anchor)

    assert output.mosaic.shape == anchor.shape
    assert torch.allclose(output.gain, torch.tensor([2.0]))
    assert torch.allclose(output.mosaic, anchor)
    assert torch.allclose(output.rgb48, torch.full_like(proraw, 1.0))


def test_global_del_tm_honors_inverse_tone_curve_before_gain():
    proraw = torch.full((1, 3, 4, 4), 0.25)
    anchor = torch.full((1, 1, 2, 2), 0.5)

    def inverse_tone(x: torch.Tensor) -> torch.Tensor:
        return x.square()

    output = baselines.global_del_tm_baseline(proraw, anchor, inverse_tone_curve=inverse_tone)

    # Inverse tone reduces to 0.0625; gain should lift by 8x (clamped to max_gain=4.0).
    assert torch.allclose(output.gain, torch.tensor([4.0]))
    assert torch.allclose(output.mosaic, quad_bayer_forward(output.rgb48))
    assert torch.all(output.rgb48 >= 0)


def test_global_del_tm_respects_gain_percentile_and_clamps():
    proraw = torch.ones((1, 3, 4, 4))
    anchor = torch.tensor([[[[1.0, 8.0], [8.0, 8.0]]]])

    output = baselines.global_del_tm_baseline(
        proraw, anchor, gain_percentile=0.75, min_gain=0.5, max_gain=6.0
    )

    # Ratios = [1,8,8,8]; 75th percentile is 8, but 6.0 clamp wins.
    assert torch.allclose(output.gain, torch.tensor([6.0]))
    assert torch.allclose(output.mosaic.mean(), torch.tensor(6.75))
