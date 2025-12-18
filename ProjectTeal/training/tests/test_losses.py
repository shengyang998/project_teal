import torch

from ProjectTeal.training.losses import (
    anchor_charbonnier_loss,
    gain_regularization,
    gradient_detail_loss,
    quad_bayer_forward,
)


def test_quad_bayer_forward_matches_expected_pattern():
    rgb = torch.zeros(1, 3, 4, 4)
    rgb[:, 0] = 1.0  # red plane
    rgb[:, 1] = 2.0  # green plane
    rgb[:, 2] = 4.0  # blue plane

    channel_scale = torch.tensor([2.0, 1.0, 0.5])
    mosaiced = quad_bayer_forward(rgb, channel_scale=channel_scale)

    expected = torch.full((1, 1, 2, 2), 2.0)
    assert torch.allclose(mosaiced, expected)


def test_anchor_charbonnier_loss_uses_forward_operator():
    rgb = torch.ones(1, 3, 4, 4)
    target = quad_bayer_forward(rgb)

    offset_target = target + 0.1
    loss = anchor_charbonnier_loss(rgb, offset_target, eps=1e-6)

    expected = torch.sqrt(torch.tensor(0.1**2 + 1e-6**2))
    assert torch.isclose(loss, expected)


def test_gradient_detail_loss_applies_global_render():
    pred = torch.zeros(1, 3, 2, 2)
    pred[:, 0, :, 1] = 1.0  # horizontal edge in red channel
    reference = torch.zeros_like(pred)

    ccm = torch.diag(torch.tensor([2.0, 1.0, 1.0]))
    loss = gradient_detail_loss(pred, reference, ccm=ccm)

    # Two horizontal gradients of magnitude 2 across batches/channels/spatial locations
    expected_dx_mean = torch.tensor(2.0 / 3)
    assert torch.isclose(loss, expected_dx_mean)


def test_gain_regularization_penalizes_smoothness_and_range():
    base = torch.tensor([[[[0.4, 1.4], [1.0, 2.5]]]])
    gain = base.repeat(1, 3, 1, 1)

    total, components = gain_regularization(gain, min_gain=0.5, max_gain=2.0)

    dx = gain[:, :, :, 1:] - gain[:, :, :, :-1]
    dy = gain[:, :, 1:, :] - gain[:, :, :-1, :]
    expected_smoothness = dx.abs().mean() + dy.abs().mean()
    expected_range = (
        torch.clamp_min(0.5 - gain, 0) + torch.clamp_min(gain - 2.0, 0)
    ).mean()
    expected_total = expected_smoothness + expected_range

    assert torch.isclose(components["smoothness"], expected_smoothness)
    assert torch.isclose(components["range"], expected_range)
    assert torch.isclose(total, expected_total)
