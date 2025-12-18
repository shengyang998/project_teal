import pytest

torch = pytest.importorskip("torch")

from ProjectTeal.training.models import GainFieldUNet, GainFieldUNetConfig
from ProjectTeal.training.models.gain_field_unet import trace_for_coreml


def test_output_shapes():
    config = GainFieldUNetConfig(in_channels=6, base_channels=16, gain_channels=3, detail_channels=3)
    model = GainFieldUNet(config)
    x = torch.randn(2, config.in_channels, 64, 64)

    outputs = model(x)

    gain = outputs["gain"]
    detail = outputs["detail"]

    assert gain.shape == (2, config.gain_channels, 8, 8)
    assert detail.shape == (2, config.detail_channels, 64, 64)


def test_export_inputs_shape_matches_config():
    config = GainFieldUNetConfig()
    model = GainFieldUNet(config)
    example, gain_out = model.export_coreml_inputs((128, 256))

    assert example.shape == (1, config.in_channels, 128, 256)
    assert gain_out.shape == (1, config.gain_channels, 32, 64)


def test_trace_for_coreml_preserves_output_shapes():
    config = GainFieldUNetConfig(base_channels=8)
    model = GainFieldUNet(config)

    traced, example, output_shapes = trace_for_coreml(model, image_size=(64, 96))

    assert example.shape == (1, config.in_channels, 64, 96)
    assert output_shapes["gain"] == torch.Size([1, config.gain_channels, 16, 24])
    assert output_shapes["detail"] == torch.Size([1, config.detail_channels, 64, 96])

    traced_outputs = traced(example)
    assert traced_outputs["gain"].shape == output_shapes["gain"]
    assert traced_outputs["detail"].shape == output_shapes["detail"]
