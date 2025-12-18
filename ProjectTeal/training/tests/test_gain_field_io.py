import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from ProjectTeal.training.models import (
    build_model_inputs,
    compose_linear_prediction,
    upsample_guidance,
)


def test_upsample_guidance_matches_target_shape():
    raw = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(1, 3, 2, 2)
    upsampled = upsample_guidance(raw, target_size=(4, 6))

    assert upsampled.shape == (1, 3, 4, 6)
    # Bilinear upsample preserves low-frequency DC component
    assert torch.isclose(upsampled.mean(), raw.mean(), atol=1e-3)


def test_build_model_inputs_concatenates_and_upsamples():
    pro_raw48 = torch.ones(1, 3, 4, 4)
    raw12 = torch.zeros(1, 3, 2, 2)
    raw12[:, :, 1, 1] = 2.0

    stacked = build_model_inputs(pro_raw48, raw12)

    assert stacked.shape == (1, 6, 4, 4)
    guidance = stacked[:, 3:, :, :]
    expected_guidance = F.interpolate(raw12, size=(4, 4), mode="bilinear", align_corners=False)
    assert torch.allclose(guidance, expected_guidance)


def test_compose_linear_prediction_applies_gain_and_residual():
    pro_raw48 = torch.ones(1, 3, 4, 4)
    gain = torch.full((1, 3, 2, 2), 0.5)
    detail = torch.full((1, 3, 4, 4), 0.25)

    output = compose_linear_prediction(pro_raw48, gain, detail)

    # Upsampled gain is 0.5 everywhere; expect max(0, 1*0.5 + 0.25) = 0.75
    assert torch.allclose(output, torch.full_like(output, 0.75))
