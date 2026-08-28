from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


class VisionTokenAdapter(nn.Module):
    """Project pooled P3/P4 features into a bounded visual-token sequence.

    The adapter is intentionally standalone: the current text checkpoint has
    no visual-token training contract, so the chat CLI uses the explicit text
    summary adapter below unless a future multimodal LM checkpoint opts in.
    """

    def __init__(self, p3_channels: int, p4_channels: int, d_model: int, num_tokens: int = 8):
        super().__init__()
        if min(p3_channels, p4_channels, d_model, num_tokens) <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.num_tokens = int(num_tokens)
        self.projection = nn.Sequential(
            nn.Linear(p3_channels + p4_channels, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * self.num_tokens),
        )

    def forward(self, features: Mapping[str, torch.Tensor], *, detach: bool = True) -> torch.Tensor:
        for level in ("P3", "P4"):
            if level not in features:
                raise ValueError(f"visual token adapter requires {level}")
        pooled = torch.cat((features["P3"].mean(dim=(-2, -1)), features["P4"].mean(dim=(-2, -1))), dim=-1)
        if detach:
            pooled = pooled.detach()
        return self.projection(pooled).view(pooled.shape[0], self.num_tokens, -1)


def vision_result_to_context(result: Mapping[str, Any]) -> str:
    """Create an explicit, inspectable text context for the legacy text LM."""
    detections = result.get("detections", [])
    detection_text = "无检测框"
    if detections:
        entries = []
        for item in detections:
            name = item.get("class_name", f"class_{item.get('class_id', 0)}")
            entries.append(f"{name} score={float(item.get('score', 0.0)):.3f} box={item.get('box_xyxy')}")
        detection_text = "; ".join(entries)
    segmentation = result.get("segmentation", {})
    classes = segmentation.get("classes", []) if isinstance(segmentation, Mapping) else []
    class_text = ", ".join(str(item) for item in classes) if classes else "无前景类别"
    return (
        "[视觉分析上下文] "
        f"图像尺寸={result.get('image_size')}; 检测={detection_text}; "
        f"分割前景类别={class_text}。"
    )
