from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn

from .backbone import SharedVisionBackbone
from .config import VisionConfig
from .detect_head import DetectionHead
from .neck import SharedMultiScaleNeck
from .segment_head import LightweightSegmentationDecoder


class DefectVisionModel(nn.Module):
    """Shared visual trunk with independent detection and segmentation heads."""

    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        self.config = config or VisionConfig()
        self.config.validate()
        use_p5 = self.config.resolve_use_p5()
        if "P5" in self.config.detect_levels and not use_p5:
            raise ValueError("P5 detection is configured but use_p5 resolves to false")

        self.backbone = SharedVisionBackbone(
            channels=self.config.backbone_channels,
            depths=self.config.backbone_depths,
            use_p5=use_p5,
        )
        self.neck = SharedMultiScaleNeck(
            in_channels=self.config.backbone_channels,
            out_channels=self.config.neck_channels,
            use_p5=use_p5,
        )
        neck_channel_map = {
            level: self.config.neck_channels[index]
            for index, level in enumerate(("P1", "P2", "P3", "P4", "P5"))
        }
        self.detection_head = DetectionHead(
            in_channels={level: neck_channel_map[level] for level in self.config.detect_levels},
            num_classes=self.config.num_classes,
            levels=self.config.detect_levels,
            reg_max=self.config.reg_max,
        )
        self.segmentation_head = LightweightSegmentationDecoder(
            feature_channels={level: neck_channel_map[level] for level in ("P1", "P2", "P3", "P4")},
            num_classes=self.config.segmentation_classes,
            output_size=self.config.input_size,
        )
        self.resolved_use_p5 = use_p5

    def forward(
        self,
        images: torch.Tensor,
        *,
        tasks: Sequence[str] = ("detection", "segmentation"),
        return_features: bool = False,
    ) -> Dict[str, Any]:
        requested = set(tasks)
        allowed = {"detection", "segmentation"}
        unknown = requested.difference(allowed)
        if unknown:
            raise ValueError(f"unknown vision tasks: {sorted(unknown)}")
        if not requested:
            raise ValueError("at least one vision task is required")
        features = self.neck(self.backbone(images))
        result: Dict[str, Any] = {}
        if "detection" in requested:
            result["detection"] = self.detection_head(features)
        if "segmentation" in requested:
            result["segmentation"] = self.segmentation_head(features)
        if return_features:
            result["features"] = features
        return result

    def freeze_for_stage(self, stage: str) -> None:
        """Apply the issue's few-shot stage ownership rules."""
        stage = stage.lower().strip()
        if stage not in {"f0", "f1", "f2", "f3", "f4", "detection", "segmentation", "joint"}:
            raise ValueError("stage must be one of f0/f1/f2/f3/f4/detection/segmentation/joint")
        for parameter in self.parameters():
            parameter.requires_grad = True

        if stage == "f0":
            for parameter in self.parameters():
                parameter.requires_grad = False
        elif stage in {"f1", "detection"}:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.detection_head.reg_towers.parameters():
                parameter.requires_grad = False
            for parameter in self.detection_head.reg_preds.parameters():
                parameter.requires_grad = False
        elif stage == "f3" or stage == "segmentation":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.neck.parameters():
                parameter.requires_grad = False
            for parameter in self.detection_head.parameters():
                parameter.requires_grad = False
        elif stage == "f2":
            # F2 is the controlled unfreeze stage; callers can assign a lower
            # learning rate to regression parameters in their optimizer.
            for parameter in self.detection_head.reg_towers.parameters():
                parameter.requires_grad = True
            for parameter in self.detection_head.reg_preds.parameters():
                parameter.requires_grad = True
        elif stage in {"f4", "joint"}:
            pass

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def save_vision_checkpoint(
    path: str | Path,
    model: DefectVisionModel,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": "chatdemo-vision-v1",
        "vision_config": model.config.to_dict(),
        "class_names": list(model.config.class_names),
        "model_state": model.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, target)
    return target


def load_vision_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[DefectVisionModel, VisionConfig, Dict[str, Any]]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"vision checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("vision checkpoint must contain a mapping payload")
    config_payload = payload.get("vision_config")
    if not isinstance(config_payload, dict):
        raise ValueError("vision checkpoint is missing vision_config; refusing implicit architecture inference")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("vision checkpoint is missing model_state")
    config = VisionConfig.from_dict(config_payload)
    model = DefectVisionModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, payload


def initialize_vision_model(
    config: VisionConfig,
    *,
    mode: str = "random",
    checkpoint: str | Path | None = None,
    mapping: str | Path | None = None,
) -> DefectVisionModel:
    """Create a model under an explicit initialization policy.

    Random initialization is intentionally opt-in for development and unit
    tests.  Production/few-shot modes require a real checkpoint and never
    silently fall back to random weights.
    """
    normalized = mode.strip().lower()
    if normalized == "random":
        return DefectVisionModel(config)
    if checkpoint is None:
        raise ValueError(f"vision_init.mode={normalized} requires an explicit checkpoint")
    if normalized == "project_base":
        model, loaded_config, _ = load_vision_checkpoint(checkpoint)
        if loaded_config.to_dict() != config.to_dict():
            raise ValueError("project_base checkpoint config does not exactly match the requested vision config")
        return model
    if normalized == "yolov8_transfer":
        if mapping is None:
            raise ValueError("yolov8_transfer requires an explicit target-to-source mapping JSON")
        model = DefectVisionModel(config)
        from .transfer import load_explicit_state_mapping
        return load_explicit_state_mapping(model, checkpoint, mapping)
    raise ValueError(f"unsupported vision initialization mode: {mode}")
