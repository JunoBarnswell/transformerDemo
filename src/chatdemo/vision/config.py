from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


@dataclass
class VisionConfig:
    input_size: int = 640
    num_classes: int = 1
    class_names: List[str] = field(default_factory=lambda: ["defect"])
    segmentation_classes: int = 2
    backbone_channels: Tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)
    backbone_depths: Tuple[int, int, int, int, int] = (1, 1, 2, 2, 1)
    neck_channels: Tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)
    use_p5: bool | str = "auto"
    detect_levels: Tuple[str, ...] = ("P2", "P3", "P4")
    reg_max: int = 16
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.7
    max_detections: int = 100
    tile_threshold_px: float = 4.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionConfig":
        payload = dict(data.get("vision", data))
        for key in ("backbone_channels", "backbone_depths", "neck_channels"):
            if key in payload:
                payload[key] = tuple(int(value) for value in payload[key])
        if "detect_levels" in payload:
            payload["detect_levels"] = tuple(str(value) for value in payload["detect_levels"])
        if "class_names" in payload:
            payload["class_names"] = [str(value) for value in payload["class_names"]]
        result = cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})
        result.validate()
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisionConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"vision config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError("vision config must contain a YAML mapping")
        return cls.from_dict(payload)

    def validate(self) -> None:
        if self.input_size != 640:
            raise ValueError("the public vision contract requires a fixed 640x640 input and coordinate space")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        if len(self.backbone_channels) != 5 or len(self.backbone_depths) != 5 or len(self.neck_channels) != 5:
            raise ValueError("backbone and neck channel/depth settings must contain P1..P5 values")
        if any(value <= 0 for value in (*self.backbone_channels, *self.backbone_depths, *self.neck_channels)):
            raise ValueError("backbone and neck widths/depths must be positive")
        if self.segmentation_classes < 2:
            raise ValueError("segmentation_classes must include background and one defect class")
        if self.reg_max <= 0:
            raise ValueError("reg_max must be positive")
        if not self.detect_levels or any(level not in {"P2", "P3", "P4", "P5"} for level in self.detect_levels):
            raise ValueError("detect_levels must contain P2/P3/P4/P5 names")
        if self.use_p5 not in (True, False, "auto"):
            raise ValueError("use_p5 must be true, false, or auto")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 < self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in (0, 1]")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")

    def resolve_use_p5(self, max_object_area_fraction: float | None = None) -> bool:
        """Resolve ``auto`` without silently disabling large-object support."""
        if self.use_p5 is True:
            return True
        if self.use_p5 is False:
            return False
        if max_object_area_fraction is None:
            return True
        if not 0.0 <= max_object_area_fraction <= 1.0:
            raise ValueError("max_object_area_fraction must be in [0, 1]")
        return max_object_area_fraction >= 0.05

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("backbone_channels", "backbone_depths", "neck_channels", "detect_levels"):
            result[key] = list(result[key])
        return result
