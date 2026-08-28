from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct


class SharedMultiScaleNeck(nn.Module):
    """Lightweight PAN/FPN with a high-resolution P2 output."""

    def __init__(
        self,
        in_channels: Sequence[int] = (16, 32, 64, 128, 256),
        out_channels: Sequence[int] = (16, 32, 64, 128, 256),
        use_p5: bool = True,
    ):
        super().__init__()
        if len(in_channels) != 5 or len(out_channels) != 5:
            raise ValueError("neck channels must contain P1..P5 values")
        self.in_channels = tuple(int(value) for value in in_channels)
        self.out_channels = tuple(int(value) for value in out_channels)
        self.use_p5 = bool(use_p5)

        self.p1_proj = ConvBNAct(self.in_channels[0], self.out_channels[0], 1)
        self.p2_lat = ConvBNAct(self.in_channels[1], self.out_channels[1], 1)
        self.p3_lat = ConvBNAct(self.in_channels[2], self.out_channels[2], 1)
        self.p4_lat = ConvBNAct(self.in_channels[3], self.out_channels[3], 1)
        self.p4_refine = ConvBNAct(self.out_channels[3], self.out_channels[3], 3)
        self.p3_refine = ConvBNAct(self.out_channels[2], self.out_channels[2], 3)
        self.p2_refine = ConvBNAct(self.out_channels[1], self.out_channels[1], 3)
        self.p4_to_p3 = ConvBNAct(self.out_channels[3], self.out_channels[2], 1)
        self.p3_to_p2 = ConvBNAct(self.out_channels[2], self.out_channels[1], 1)

        self.p2_down = ConvBNAct(self.out_channels[1], self.out_channels[2], 3, stride=2)
        self.p3_down = ConvBNAct(self.out_channels[2], self.out_channels[3], 3, stride=2)

        if self.use_p5:
            self.p5_lat = ConvBNAct(self.in_channels[4], self.out_channels[4], 1)
            self.p5_refine = ConvBNAct(self.out_channels[4], self.out_channels[4], 3)
            self.p5_to_p4 = ConvBNAct(self.out_channels[4], self.out_channels[3], 1)
            self.p4_down = ConvBNAct(self.out_channels[3], self.out_channels[4], 3, stride=2)

    @staticmethod
    def _up_add(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return low + F.interpolate(high, size=low.shape[-2:], mode="nearest")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = {"P1", "P2", "P3", "P4"}
        missing = required.difference(features)
        if missing:
            raise ValueError(f"backbone features missing: {sorted(missing)}")

        p1 = self.p1_proj(features["P1"])
        p2 = self.p2_lat(features["P2"])
        p3 = self.p3_lat(features["P3"])
        p4 = self.p4_lat(features["P4"])
        if self.use_p5:
            if "P5" not in features:
                raise ValueError("P5 is required when use_p5=True")
            p5 = self.p5_lat(features["P5"])
            p4 = self.p4_refine(self._up_add(self.p5_to_p4(p5), p4))
        else:
            p4 = self.p4_refine(p4)
        p3 = self.p3_refine(self._up_add(self.p4_to_p3(p4), p3))
        p2 = self.p2_refine(self._up_add(self.p3_to_p2(p3), p2))

        p3 = self.p3_refine(p3 + self.p2_down(p2))
        p4 = self.p4_refine(p4 + self.p3_down(p3))
        output = {"P1": p1, "P2": p2, "P3": p3, "P4": p4}
        if self.use_p5:
            p5 = self.p5_refine(p5 + self.p4_down(p4))
            output["P5"] = p5
        return output
