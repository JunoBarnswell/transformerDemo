from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .coordinates import canonical_to_original, original_to_canonical
from .postprocess import nms
from .types import Detection


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("tile window must have positive area")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def tile_windows(width: int, height: int, *, tile_size: int = 640, overlap: float = 0.2) -> list[TileWindow]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0,1)")
    if width <= tile_size and height <= tile_size:
        return [TileWindow(0, 0, width, height)]
    stride = max(1, round(tile_size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        values = list(range(0, max(1, length - tile_size + 1), stride))
        last = max(0, length - tile_size)
        if not values or values[-1] != last:
            values.append(last)
        return sorted(set(values))

    windows = []
    for y0 in starts(height):
        for x0 in starts(width):
            windows.append(TileWindow(x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size)))
    return windows


def should_use_tiles(
    image_width: int,
    image_height: int,
    *,
    estimated_defect_short_side: float | None,
    threshold_px: float = 4.0,
    model_size: int = 640,
) -> bool:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if threshold_px <= 0 or model_size <= 0:
        raise ValueError("threshold_px and model_size must be positive")
    if image_width <= model_size and image_height <= model_size:
        return False
    if estimated_defect_short_side is None:
        raise ValueError("auto tile selection requires an estimated defect size")
    if estimated_defect_short_side < 0:
        raise ValueError("estimated_defect_short_side must be non-negative")
    return estimated_defect_short_side < threshold_px


def merge_tile_detections(
    tile_results: Sequence[tuple[TileWindow, Sequence[Detection]]],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
) -> list[Detection]:
    """Map tile-local canonical boxes to one global canonical coordinate space."""
    mapped: list[Detection] = []
    for window, detections in tile_results:
        for detection in detections:
            local_original = canonical_to_original(
                [detection.box_xyxy], window.width, window.height
            )[0]
            global_original = local_original.clone()
            global_original[[0, 2]] += window.x0
            global_original[[1, 3]] += window.y0
            global_canonical = original_to_canonical(
                [global_original.tolist()], image_width, image_height
            )[0]
            box = tuple(float(value) for value in global_canonical.tolist())
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            mapped.append(Detection(detection.class_id, detection.score, box, detection.class_name))

    output: list[Detection] = []
    for class_id in sorted({item.class_id for item in mapped}):
        class_detections = [item for item in mapped if item.class_id == class_id]
        boxes = torch.tensor([item.box_xyxy for item in class_detections], dtype=torch.float32)
        scores = torch.tensor([item.score for item in class_detections], dtype=torch.float32)
        keep = nms(boxes, scores, iou_threshold=iou_threshold, max_detections=max_detections)
        output.extend(class_detections[index] for index in keep.tolist())
    output.sort(key=lambda item: item.score, reverse=True)
    return output[:max_detections]


def merge_tile_probabilities(
    tile_results: Sequence[tuple[TileWindow, torch.Tensor]],
    *,
    image_width: int,
    image_height: int,
    num_classes: int,
) -> torch.Tensor:
    """Average overlapping tile probabilities and return an original-size mask."""
    if num_classes < 2:
        raise ValueError("num_classes must include background")
    if not tile_results:
        raise ValueError("at least one tile probability map is required")
    first = tile_results[0][1]
    if first.ndim != 3 or first.shape[0] != num_classes:
        raise ValueError("tile probabilities must have shape [C,H,W]")
    device = first.device
    sums = torch.zeros((num_classes, image_height, image_width), dtype=first.dtype, device=device)
    counts = torch.zeros((1, image_height, image_width), dtype=first.dtype, device=device)
    for window, probabilities in tile_results:
        if probabilities.ndim != 3 or probabilities.shape[0] != num_classes or tuple(probabilities.shape[-2:]) != (window.height, window.width):
            raise ValueError("tile probability shape does not match its window")
        sums[:, window.y0:window.y1, window.x0:window.x1] += probabilities
        counts[:, window.y0:window.y1, window.x0:window.x1] += 1.0
    return (sums / counts.clamp_min(1.0)).argmax(dim=0)
