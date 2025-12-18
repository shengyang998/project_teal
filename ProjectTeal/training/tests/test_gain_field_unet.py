import torch

from ProjectTeal.training.models import GainFieldUNet, GainFieldUNetConfig


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
