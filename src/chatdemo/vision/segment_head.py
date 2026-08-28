from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNAct


def _feature_channels(feature_channels: int | Mapping[str, int] | Sequence[int]) -> Dict[str, int]:
    if isinstance(feature_channels, int):
        return {"P1": feature_channels, "P2": feature_channels, "P3": feature_channels, "P4": feature_channels}
    if isinstance(feature_channels, Mapping):
        missing = [level for level in ("P1", "P2", "P3", "P4") if level not in feature_channels]
        if missing:
            raise ValueError(f"segmentation feature channels missing: {missing}")
        return {level: int(feature_channels[level]) for level in ("P1", "P2", "P3", "P4")}
    values = list(feature_channels)
    if len(values) != 4:
        raise ValueError("segmentation feature channel sequence must contain P1..P4")
    return {level: int(value) for level, value in zip(("P1", "P2", "P3", "P4"), values)}


class LightweightSegmentationDecoder(nn.Module):
    """Independent P1/P2/P3/P4 decoder for full-image semantic logits."""

    def __init__(
        self,
        feature_channels: int | Mapping[str, int] | Sequence[int],
        num_classes: int = 2,
        hidden_channels: int = 64,
        output_size: int = 640,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("segmentation requires background plus at least one class")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        channels = _feature_channels(feature_channels)
        self.num_classes = int(num_classes)
        self.output_size = int(output_size)
        self.p1_align = ConvBNAct(channels["P1"], hidden_channels, 1)
        self.p2_align = ConvBNAct(channels["P2"], hidden_channels, 1)
        self.p3_align = ConvBNAct(channels["P3"], hidden_channels, 1)
        self.p4_align = ConvBNAct(channels["P4"], hidden_channels, 1)
        self.p4_refine = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.p3_refine = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.p2_refine = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.p1_refine = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.classifier = nn.Conv2d(hidden_channels, self.num_classes, 1)

    @staticmethod
    def _up_add(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return low + F.interpolate(high, size=low.shape[-2:], mode="nearest")

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = ("P1", "P2", "P3", "P4")
        missing = [level for level in required if level not in features]
        if missing:
            raise ValueError(f"features missing segmentation levels: {missing}")
        p4 = self.p4_refine(self.p4_align(features["P4"]))
        p3 = self.p3_refine(self._up_add(p4, self.p3_align(features["P3"])))
        p2 = self.p2_refine(self._up_add(p3, self.p2_align(features["P2"])))
        p1 = self.p1_refine(self._up_add(p2, self.p1_align(features["P1"])))
        logits = self.classifier(p1)
        return F.interpolate(logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
