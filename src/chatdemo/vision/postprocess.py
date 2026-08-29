from __future__ import annotations

from typing import Sequence

import torch

from .coordinates import box_iou, model_to_canonical
from .detect_head import decode_raw_predictions
from .types import Detection, ImageTransform


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.7, max_detections: int = 300) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("nms expects boxes [N,4] and scores [N]")
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() and len(kept) < max_detections:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        overlap = box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= iou_threshold]
    return torch.stack(kept)


def decode_detections(raw_outputs, transforms: Sequence[ImageTransform], *, confidence_threshold: float = 0.25, nms_threshold: float = 0.7, max_detections: int = 300, class_names: Sequence[str] | None = None) -> list[list[Detection]]:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")
    decoded = decode_raw_predictions(raw_outputs)
    if len(transforms) != decoded.boxes.shape[0]:
        raise ValueError("one transform is required per batch item")
    results = []
    for batch_index, transform in enumerate(transforms):
        scores, labels = decoded.scores[batch_index].max(dim=-1)
        candidate = scores >= confidence_threshold
        detections: list[Detection] = []
        for class_id in labels[candidate].unique(sorted=True).tolist():
            indices = torch.where(candidate & (labels == class_id))[0]
            keep = nms(decoded.boxes[batch_index, indices], scores[indices], nms_threshold, max_detections)
            boxes = model_to_canonical(decoded.boxes[batch_index, indices[keep]], transform)
            for box, score in zip(boxes.tolist(), scores[indices[keep]].tolist()):
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                name = class_names[class_id] if class_names is not None and class_id < len(class_names) else None
                detections.append(Detection(int(class_id), float(score), tuple(box), name))
        detections.sort(key=lambda item: item.score, reverse=True)
        results.append(detections[:max_detections])
    return results
