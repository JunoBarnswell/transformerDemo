"""Versioned native YOLOv8 v8.4.0 detection components."""

from .config import VisionConfig
from .coordinates import CANONICAL_SIZE, canonical_to_original, model_to_canonical, original_to_canonical
from .detect_head import DetectionHead, TaskAlignedAssigner
from .losses import DetectionLoss
from .model import DefectVisionModel
from .postprocess import decode_detections, nms
from .types import Detection, ImageTransform, VisionSample

__all__ = [
    "DefectVisionModel",
    "DetectionHead",
    "DetectionLoss",
    "Detection",
    "ImageTransform",
    "CANONICAL_SIZE",
    "TaskAlignedAssigner",
    "VisionConfig",
    "VisionSample",
    "canonical_to_original",
    "decode_detections",
    "model_to_canonical",
    "nms",
    "original_to_canonical",
]
