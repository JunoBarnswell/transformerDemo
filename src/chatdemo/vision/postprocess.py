from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .coordinates import box_iou, model_to_canonical
from .detect_head import DecodedPredictions, decode_raw_predictions
from .preprocess import load_image
from .types import Detection, ImageTransform


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.7, max_detections: int = 100) -> torch.Tensor:
    """Class-local greedy NMS returning indices in descending score order."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("nms expects boxes [N,4] and scores [N]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0,1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < max_detections:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        overlaps = box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return torch.stack(keep).long()


def decode_detections(
    raw_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    transforms: Sequence[ImageTransform],
    *,
    reg_max: int,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.7,
    max_detections: int = 100,
    class_names: Sequence[str] | None = None,
    levels: Sequence[str] | None = None,
) -> list[list[Detection]]:
    """Decode detector outputs and convert boxes to canonical coordinates."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1]")
    decoded: DecodedPredictions = decode_raw_predictions(raw_outputs, reg_max=reg_max, levels=levels)
    if len(transforms) != decoded.boxes.shape[0]:
        raise ValueError("one ImageTransform is required per batch item")

    results: list[list[Detection]] = []
    for batch_index, transform in enumerate(transforms):
        boxes = decoded.boxes[batch_index]
        scores, labels = decoded.scores[batch_index].max(dim=-1)
        candidate = scores >= confidence_threshold
        detections: list[Detection] = []
        for class_id in labels[candidate].unique(sorted=True).tolist():
            class_mask = candidate & (labels == class_id)
            indices = torch.where(class_mask)[0]
            keep_local = nms(boxes[indices], scores[indices], nms_threshold, max_detections)
            kept_indices = indices[keep_local]
            canonical_boxes = model_to_canonical(boxes[kept_indices], transform)
            for box, score in zip(canonical_boxes.tolist(), scores[kept_indices].tolist()):
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = class_names[class_id] if class_names is not None and class_id < len(class_names) else None
                detections.append(Detection(int(class_id), float(score), (x1, y1, x2, y2), class_name))
        detections.sort(key=lambda item: item.score, reverse=True)
        results.append(detections[:max_detections])
    return results


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (239, 68, 68),
        (245, 158, 11),
        (34, 197, 94),
        (59, 130, 246),
        (168, 85, 247),
        (14, 165, 233),
    )
    return palette[class_id % len(palette)]


def save_mask_png(mask: torch.Tensor, path: str | Path) -> Path:
    """Save class-index mask at its existing original-image resolution."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to save visual outputs") from exc
    if mask.ndim != 2:
        raise ValueError("mask must have shape [H,W]")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = mask.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(array, mode="L").save(target)
    return target


def render_annotated_image(
    image: object,
    mask: torch.Tensor | None,
    detections: Sequence[Detection],
    path: str | Path,
    *,
    mask_alpha: int = 90,
) -> Path:
    """Render original-size mask overlay and detector-owned boxes."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to render visual outputs") from exc
    if not 0 <= mask_alpha <= 255:
        raise ValueError("mask_alpha must be in [0,255]")
    base = load_image(image).convert("RGBA")
    if mask is not None:
        if mask.ndim != 2 or tuple(mask.shape) != (base.height, base.width):
            raise ValueError("mask must match the original image resolution")
        mask_array = mask.detach().cpu().to(torch.long).numpy()
        overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)
        for class_id in np.unique(mask_array):
            if int(class_id) <= 0:
                continue
            color = color_for_class(int(class_id) - 1)
            pixels = mask_array == class_id
            overlay[pixels, :3] = color
            overlay[pixels, 3] = mask_alpha
        base = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))

    draw = ImageDraw.Draw(base)
    sx = base.width / 640.0
    sy = base.height / 640.0
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        coordinates = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        color = color_for_class(detection.class_id)
        draw.rectangle(coordinates, outline=color + (255,), width=max(1, round(min(base.width, base.height) / 320)))
        label = detection.class_name or f"class_{detection.class_id}"
        draw.text((coordinates[0], max(0, coordinates[1] - 12)), f"{label} {detection.score:.2f}", fill=color + (255,))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(target)
    return target
