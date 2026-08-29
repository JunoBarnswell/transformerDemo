from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import C2f, Concat, Conv, SPPF
from .config import VisionConfig
from .detect_head import DetectionHead


class DefectVisionModel(nn.Module):
    """Native YOLOv8 v8.4.0 detection graph.

    The module indices intentionally match the official YAML graph so an
    explicit converter can compare intermediate features and transfer shared
    weights without guessing parameter names.
    """

    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        self.config = config or VisionConfig()
        self.config.validate()
        p2 = self.config.architecture == "yolov8n-p2"
        layers: list[nn.Module] = [
            Conv(3, 16, 3, 2), Conv(16, 32, 3, 2), C2f(32, 32, 1, True),
            Conv(32, 64, 3, 2), C2f(64, 64, 2, True), Conv(64, 128, 3, 2),
            C2f(128, 128, 2, True), Conv(128, 256, 3, 2), C2f(256, 256, 1, True),
            SPPF(256, 256, 5),
            nn.Upsample(scale_factor=2, mode="nearest"), Concat(), C2f(384, 128, 1),
            nn.Upsample(scale_factor=2, mode="nearest"), Concat(), C2f(192, 64, 1),
        ]
        if p2:
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"), Concat(), C2f(96, 32, 1),
                Conv(32, 32, 3, 2), Concat(), C2f(96, 64, 1),
                Conv(64, 64, 3, 2), Concat(), C2f(192, 128, 1),
                Conv(128, 128, 3, 2), Concat(), C2f(384, 256, 1),
            ]
            detection_head = DetectionHead((32, 64, 128, 256), self.config.num_classes, self.config.input_size)
        else:
            layers += [Conv(64, 64, 3, 2), Concat(), C2f(192, 128, 1), Conv(128, 128, 3, 2), Concat(), C2f(384, 256, 1)]
            detection_head = DetectionHead((64, 128, 256), self.config.num_classes, self.config.input_size)
        layers.append(detection_head)
        self.model = nn.ModuleList(layers)
        self._p2 = p2

    @property
    def detection_head(self) -> DetectionHead:
        return self.model[-1]  # type: ignore[return-value]

    def forward(self, images: torch.Tensor, *, tasks: tuple[str, ...] = ("detection",), return_features: bool = False) -> dict[str, Any]:
        if set(tasks) != {"detection"}:
            raise ValueError("this checkpoint is detection-only; segmentation and multitask paths are not available")
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (self.config.input_size, self.config.input_size):
            raise ValueError(f"expected BCHW RGB tensors at {self.config.input_size}x{self.config.input_size}")
        y: list[torch.Tensor | None] = []
        x = images
        for index, layer in enumerate(self.model[:-1]):
            if self._p2:
                source = {
                    0: [-1], 1: [-1], 2: [-1], 3: [-1], 4: [-1], 5: [-1], 6: [-1], 7: [-1], 8: [-1], 9: [-1],
                    10: [-1], 11: [-1, 6], 12: [-1], 13: [-1], 14: [-1, 4], 15: [-1], 16: [-1], 17: [-1, 2], 18: [-1],
                    19: [-1], 20: [-1, 15], 21: [-1], 22: [-1], 23: [-1, 12], 24: [-1], 25: [-1], 26: [-1, 9], 27: [-1],
                }[index]
            else:
                source = {
                    0: [-1], 1: [-1], 2: [-1], 3: [-1], 4: [-1], 5: [-1], 6: [-1], 7: [-1], 8: [-1], 9: [-1],
                    10: [-1], 11: [-1, 6], 12: [-1], 13: [-1], 14: [-1, 4], 15: [-1], 16: [-1], 17: [-1, 12],
                    18: [-1], 19: [-1], 20: [-1, 9], 21: [-1],
                }[index]
            inputs = [x if item == -1 else y[item] for item in source]
            if any(item is None for item in inputs):
                raise RuntimeError(f"invalid graph reference at layer {index}")
            x = layer(inputs[0] if len(inputs) == 1 else inputs)
            y.append(x)
        if self._p2:
            features = (y[18], y[21], y[24], y[27])
        else:
            features = (y[15], y[18], y[21])
        if any(item is None for item in features):
            raise RuntimeError("detection graph produced missing feature")
        result: dict[str, Any] = {"detection": self.detection_head(tuple(features))}
        if return_features:
            result["features"] = tuple(features)
        return result

    def freeze_for_stage(self, stage: str) -> None:
        stage = stage.lower().strip()
        if stage not in {"baseline", "d1", "d2", "d3"}:
            raise ValueError("stage must be baseline, d1, d2, or d3")
        for parameter in self.parameters():
            parameter.requires_grad = True
        if stage == "d1":
            for parameter in self.model[:10].parameters():
                parameter.requires_grad = False
        elif stage == "d2":
            for parameter in self.model[:7].parameters():
                parameter.requires_grad = False

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def save_vision_checkpoint(path: str | Path, model: DefectVisionModel, *, extra: Mapping[str, Any] | None = None, optimizer: Any = None, scheduler: Any = None, ema_state: Mapping[str, Any] | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "chatdemo-detection-v4" if model.config.architecture.endswith("-p2") else "chatdemo-detection-v3",
        "reference": "ultralytics-v8.4.0",
        "architecture": model.config.architecture,
        "vision_config": model.config.to_dict(),
        "class_names": list(model.config.class_names),
        "model_state": model.state_dict(),
    }
    if ema_state is not None:
        payload["ema_state"] = dict(ema_state)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, target)
    return target


def load_vision_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[DefectVisionModel, VisionConfig, dict[str, Any]]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"vision checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") not in {"chatdemo-detection-v3", "chatdemo-detection-v4"}:
        raise ValueError("unsupported checkpoint format; old vision checkpoints require clean retraining")
    if payload.get("reference") != "ultralytics-v8.4.0":
        raise ValueError("checkpoint reference version is not ultralytics-v8.4.0")
    config_payload = payload.get("vision_config")
    state = payload.get("model_state")
    if not isinstance(config_payload, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain vision_config and model_state")
    config = VisionConfig.from_dict(config_payload)
    expected_format = "chatdemo-detection-v4" if config.architecture.endswith("-p2") else "chatdemo-detection-v3"
    if payload.get("format") != expected_format or payload.get("architecture") != config.architecture:
        raise ValueError("checkpoint format and architecture do not match")
    model = DefectVisionModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, payload


def initialize_vision_model(config: VisionConfig, *, mode: str = "random", checkpoint: str | Path | None = None, mapping: str | Path | None = None) -> DefectVisionModel:
    normalized = mode.strip().lower()
    if normalized == "random":
        return DefectVisionModel(config)
    if checkpoint is None:
        raise ValueError(f"vision_init.mode={normalized} requires an explicit checkpoint")
    if normalized == "project_base":
        model, loaded_config, _ = load_vision_checkpoint(checkpoint)
        if loaded_config.to_dict() != config.to_dict():
            raise ValueError("project_base checkpoint config does not exactly match requested config")
        return model
    if normalized == "yolov8_transfer":
        if mapping is None:
            raise ValueError("yolov8_transfer requires an explicit mapping JSON")
        model = DefectVisionModel(config)
        from .transfer import load_explicit_state_mapping
        return load_explicit_state_mapping(model, checkpoint, mapping)
    raise ValueError(f"unsupported vision initialization mode: {mode}")
