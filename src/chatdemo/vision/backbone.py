from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.cv2(self.cv1(x))
        return x + result if self.shortcut else result


class C2f(nn.Module):
    """Small CSP-style block with stable, explicit feature widths."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 1):
        super().__init__()
        hidden = max(out_channels // 2, 1)
        self.cv1 = ConvBNAct(in_channels, hidden * 2, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden) for _ in range(max(1, depth)))
        self.cv2 = ConvBNAct(hidden * (2 + len(self.blocks)), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SharedVisionBackbone(nn.Module):
    """Nano-sized shared trunk producing P1..P5 feature maps."""

    def __init__(
        self,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        depths: Sequence[int] = (1, 1, 2, 2, 1),
        use_p5: bool = True,
    ):
        super().__init__()
        if len(channels) != 5 or len(depths) != 5:
            raise ValueError("backbone channels and depths must contain P1..P5 values")
        if any(value <= 0 for value in channels):
            raise ValueError("backbone channels must be positive")
        self.channels = tuple(int(value) for value in channels)
        self.depths = tuple(int(value) for value in depths)
        self.use_p5 = bool(use_p5)

        self.stem = ConvBNAct(3, self.channels[0], 3, stride=2)
        self.p1 = C2f(self.channels[0], self.channels[0], self.depths[0])
        self.down2 = ConvBNAct(self.channels[0], self.channels[1], 3, stride=2)
        self.p2 = C2f(self.channels[1], self.channels[1], self.depths[1])
        self.down3 = ConvBNAct(self.channels[1], self.channels[2], 3, stride=2)
        self.p3 = C2f(self.channels[2], self.channels[2], self.depths[2])
        self.down4 = ConvBNAct(self.channels[2], self.channels[3], 3, stride=2)
        self.p4 = C2f(self.channels[3], self.channels[3], self.depths[3])
        self.down5 = ConvBNAct(self.channels[3], self.channels[4], 3, stride=2)
        self.p5 = C2f(self.channels[4], self.channels[4], self.depths[4])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected BCHW RGB tensor, got {tuple(x.shape)}")
        p1 = self.p1(self.stem(x))
        p2 = self.p2(self.down2(p1))
        p3 = self.p3(self.down3(p2))
        p4 = self.p4(self.down4(p3))
        features = {"P1": p1, "P2": p2, "P3": p3, "P4": p4}
        if self.use_p5:
            features["P5"] = self.p5(self.down5(p4))
        return features
