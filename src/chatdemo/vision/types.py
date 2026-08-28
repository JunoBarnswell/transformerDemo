from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ImageTransform:
    """Geometry metadata for one letterboxed image.

    ``scale`` and padding describe the model-input coordinate system.  Public
    detections are never emitted in this coordinate system; they are converted
    to canonical 640 coordinates before leaving the inference boundary.
    """

    orig_w: int
    orig_h: int
    scale: float
    pad_x: float
    pad_y: float
    input_size: int = 640

    def __post_init__(self) -> None:
        if self.orig_w <= 0 or self.orig_h <= 0:
            raise ValueError("original image dimensions must be positive")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be a finite positive number")
        if not isfinite(self.pad_x) or not isfinite(self.pad_y):
            raise ValueError("padding must be finite")

    @property
    def resized_w(self) -> int:
        return max(1, round(self.orig_w * self.scale))

    @property
    def resized_h(self) -> int:
        return max(1, round(self.orig_h * self.scale))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orig_w": self.orig_w,
            "orig_h": self.orig_h,
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "input_size": self.input_size,
        }


@dataclass(frozen=True)
class Detection:
    """One detector-owned prediction in canonical 640 coordinates."""

    class_id: int
    score: float
    box_xyxy: Tuple[float, float, float, float]
    class_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be a finite number in [0, 1]")
        if len(self.box_xyxy) != 4:
            raise ValueError("box_xyxy must contain four coordinates")
        if not all(isfinite(float(value)) for value in self.box_xyxy):
            raise ValueError("box_xyxy must contain finite coordinates")
        x1, y1, x2, y2 = self.box_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box_xyxy must have positive area")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "class_id": self.class_id,
            "score": round(float(self.score), 6),
            "box_xyxy": [round(float(value), 4) for value in self.box_xyxy],
        }
        if self.class_name is not None:
            payload["class_name"] = self.class_name
        return payload


@dataclass(frozen=True)
class VisionSample:
    """A parsed mixed-label sample.

    ``has_detection_labels`` and ``has_segmentation_label`` distinguish
    missing supervision from an explicitly negative example.  This prevents a
    partially annotated sample from silently becoming a negative sample for
    the other task.
    """

    image_path: str
    boxes: Sequence[Sequence[float]] = field(default_factory=tuple)
    labels: Sequence[int] = field(default_factory=tuple)
    mask_path: Optional[str] = None
    has_detection_labels: bool = False
    has_segmentation_label: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
