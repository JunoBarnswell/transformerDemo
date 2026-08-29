from __future__ import annotations

from typing import Sequence

import torch

from .data import VisionDataset
from .model import DefectVisionModel
from .postprocess import decode_detections


@torch.inference_mode()
def mine_hard_negative_indices(
    model: DefectVisionModel,
    dataset: VisionDataset,
    *,
    confidence_threshold: float | None = None,
) -> list[int]:
    """Find explicitly normal images that the current model falsely detects.

    The function returns source sample indices for a subsequent training round;
    it does not invent boxes or mutate labels.  A caller can crop and annotate
    the returned false-positive regions as a separate, reviewed dataset.
    """
    threshold = model.config.confidence_threshold if confidence_threshold is None else confidence_threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")
    model.eval()
    indices: list[int] = []
    for index, sample in enumerate(dataset.samples):
        if not sample.has_detection_labels or sample.boxes:
            continue
        image, target = dataset[index]
        outputs = model(image.unsqueeze(0), tasks=("detection",))
        detections = decode_detections(
            outputs["detection"],
            [target["transform"]],
            confidence_threshold=threshold,
            nms_threshold=model.config.nms_threshold,
            max_detections=model.config.max_detections,
            class_names=model.config.class_names,
        )[0]
        if detections:
            indices.append(index)
    return indices
