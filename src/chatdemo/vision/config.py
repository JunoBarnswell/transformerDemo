from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionConfig:
    """Versioned public configuration for the native YOLOv8 detector."""

    reference: str = "ultralytics-v8.4.0"
    architecture: str = "yolov8n"
    input_size: int = 640
    num_classes: int = 1
    class_names: list[str] = field(default_factory=lambda: ["defect"])
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.7
    max_detections: int = 300

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        payload = dict(data.get("vision", data))
        result = cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})
        result.validate()
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisionConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"vision config not found: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("vision config must contain a YAML mapping")
        return cls.from_dict(payload)

    def validate(self) -> None:
        if self.reference != "ultralytics-v8.4.0":
            raise ValueError("reference is fixed to ultralytics-v8.4.0")
        if self.architecture not in {"yolov8n", "yolov8n-p2"}:
            raise ValueError("architecture must be yolov8n or yolov8n-p2")
        if self.input_size != 640:
            raise ValueError("the vision contract requires a fixed 640x640 input")
        if self.num_classes <= 0 or len(self.class_names) != self.num_classes:
            raise ValueError("num_classes must equal the class_names length and be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if not 0.0 < self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in (0,1]")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")

    @property
    def detect_levels(self) -> tuple[str, ...]:
        return ("P2", "P3", "P4", "P5") if self.architecture.endswith("-p2") else ("P3", "P4", "P5")

    @property
    def reg_max(self) -> int:
        return 16

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["class_names"] = list(self.class_names)
        return result
