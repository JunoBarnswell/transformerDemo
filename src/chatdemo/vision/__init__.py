"""Shared visual defect-analysis components.

The package keeps detection and segmentation as independent task heads over a
shared visual trunk.  It intentionally does not depend on the text model so
that either visual task can be trained and evaluated on its own.
"""

from .config import VisionConfig
from .coordinates import CANONICAL_SIZE, canonical_to_original, model_to_canonical, original_to_canonical
from .detect_head import DetectionHead, TaskAlignedAssigner
from .losses import DetectionLoss, SegmentationLoss
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
    "SegmentationLoss",
    "TaskAlignedAssigner",
    "VisionConfig",
    "VisionSample",
    "canonical_to_original",
    "decode_detections",
    "model_to_canonical",
    "nms",
    "original_to_canonical",
]
