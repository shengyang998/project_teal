"""Two-head UNet-style network for gain-field and detail prediction.

The model consumes a concatenated tensor of linearized ProRAW48 RGB and
RAW12-guidance RGB (upsampled to 48MP). It produces:
- A low-resolution gain field (smooth, inverse-LTM) suitable for upsampling.
- A full-resolution residual detail map.

The architecture is kept Core ML–friendly (no custom ops; bilinear upsample) and
keeps channel counts modest to fit mobile constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GainFieldUNetConfig:
    """Configuration for the two-head UNet model."""

    in_channels: int = 6  # 3-channel ProRAW48 + 3-channel RAW12 guidance
    base_channels: int = 32
    num_downsamples: int = 3
    gain_channels: int = 3
    detail_channels: int = 3
    gain_scale: int = 4  # gain output is 1/(2^gain_scale_power) of input spatial dims


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GainFieldUNet(nn.Module):
    """Two-head UNet with low-res gain field and full-res detail outputs."""

    def __init__(self, config: GainFieldUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or GainFieldUNetConfig()

        c = self.config.base_channels
        self.enc1 = DoubleConv(self.config.in_channels, c)
        self.enc2 = DownBlock(c, c * 2)
        self.enc3 = DownBlock(c * 2, c * 4)
        self.enc4 = DownBlock(c * 4, c * 8)

        self.up3 = UpBlock(c * 8 + c * 4, c * 4)
        self.up2 = UpBlock(c * 4 + c * 2, c * 2)
        self.up1 = UpBlock(c * 2 + c, c)

        self.detail_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, self.config.detail_channels, kernel_size=1),
        )

        gain_in_channels = c * 8
        self.gain_head = nn.Sequential(
            nn.Conv2d(gain_in_channels, c * 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, self.config.gain_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the network forward and return both heads.

        Args:
            x: Tensor of shape (B, C, H, W) where C == config.in_channels.

        Returns:
            Dict with keys `gain` (B, gain_channels, H/4, W/4) and `detail`
            (B, detail_channels, H, W).
        """

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        gain = self.gain_head(e4)

        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        detail = self.detail_head(d1)

        return {"gain": gain, "detail": detail}

    @torch.no_grad()
    def export_coreml_inputs(self, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create example inputs for tracing/export with consistent shapes."""

        h, w = image_size
        example = torch.zeros(1, self.config.in_channels, h, w, dtype=torch.float32)
        gain_out = torch.zeros(1, self.config.gain_channels, h // 4, w // 4, dtype=torch.float32)
        return example, gain_out


@torch.no_grad()
def trace_for_coreml(
    model: GainFieldUNet, image_size: Tuple[int, int]
) -> Tuple[torch.jit.ScriptModule, torch.Tensor, Dict[str, torch.Size]]:
    """Trace the model for Core ML export and return shapes for outputs.

    This helper keeps export-specific logic in one place so tests can validate
    the traced graph and output tensor sizes without depending on coremltools
    during CI runs.
    """

    example, _ = model.export_coreml_inputs(image_size)
    traced = torch.jit.trace(model.eval(), example)
    outputs = traced(example)
    output_shapes = {name: tensor.shape for name, tensor in outputs.items()}
    return traced, example, output_shapes


def export_to_coreml(model: GainFieldUNet, image_size: Tuple[int, int], file_path: str) -> None:
    """Export the model to Core ML via torchscript tracing."""

    try:
        import coremltools as ct  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("coremltools is required for export") from exc

    traced, example, output_shapes = trace_for_coreml(model, image_size)
    output_specs = [
        ct.TensorType(name=name, shape=shape)
        for name, shape in output_shapes.items()
    ]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=output_specs,
    )
    mlmodel.save(file_path)
