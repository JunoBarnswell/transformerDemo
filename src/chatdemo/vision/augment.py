from __future__ import annotations

import random
from typing import Optional

import torch


class ConservativeAugment:
    """Geometry-safe augmentations for industrial few-shot samples."""

    def __init__(
        self,
        *,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.0,
        rotate90: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ):
        for name, probability in (("horizontal_flip", horizontal_flip), ("vertical_flip", vertical_flip), ("rotate90", rotate90)):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if brightness < 0 or contrast < 0:
            raise ValueError("brightness and contrast ranges must be non-negative")
        self.horizontal_flip = float(horizontal_flip)
        self.vertical_flip = float(vertical_flip)
        self.rotate90 = float(rotate90)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if image.ndim != 3 or image.shape[0] != 3 or image.shape[-1] != image.shape[-2]:
            raise ValueError("augment expects a square CHW RGB tensor")
        size = float(image.shape[-1])
        result_image = image
        result_boxes = boxes.clone()
        result_mask = mask

        if random.random() < self.horizontal_flip:
            result_image = result_image.flip(-1)
            if result_mask is not None:
                result_mask = result_mask.flip(-1)
            if result_boxes.numel():
                result_boxes[:, [0, 2]] = size - result_boxes[:, [2, 0]]

        if random.random() < self.vertical_flip:
            result_image = result_image.flip(-2)
            if result_mask is not None:
                result_mask = result_mask.flip(-2)
            if result_boxes.numel():
                result_boxes[:, [1, 3]] = size - result_boxes[:, [3, 1]]

        if random.random() < self.rotate90:
            turns = random.randint(1, 3)
            for _ in range(turns):
                result_image = torch.rot90(result_image, 1, dims=(-2, -1))
                if result_mask is not None:
                    result_mask = torch.rot90(result_mask, 1, dims=(-2, -1))
                if result_boxes.numel():
                    x1, y1, x2, y2 = result_boxes.unbind(dim=1)
                    result_boxes = torch.stack((y1, size - x2, y2, size - x1), dim=1)

        if self.brightness or self.contrast:
            brightness_delta = (random.random() * 2.0 - 1.0) * self.brightness
            contrast_factor = 1.0 + (random.random() * 2.0 - 1.0) * self.contrast
            result_image = ((result_image - 0.5) * contrast_factor + 0.5 + brightness_delta).clamp(0.0, 1.0)
        return result_image.contiguous(), result_boxes.contiguous(), result_mask.contiguous() if result_mask is not None else None


def copy_paste_mask(
    destination: torch.Tensor,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    class_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste a real source mask into a same-sized destination image.

    This is deliberately an offline data operation.  It returns the composed
    image and a class-index mask, so detection boxes can be derived from the
    resulting *training annotation* without affecting online detection output.
    """
    if destination.shape != source.shape or destination.ndim != 3 or destination.shape[0] != 3:
        raise ValueError("destination and source must be same-sized CHW RGB tensors")
    if source_mask.ndim != 2 or source_mask.shape != destination.shape[-2:]:
        raise ValueError("source_mask must match image spatial dimensions")
    if class_id <= 0:
        raise ValueError("class_id must be positive because zero is background")
    foreground = source_mask > 0
    composed = destination.clone()
    composed[:, foreground] = source[:, foreground]
    mask = torch.zeros_like(source_mask, dtype=torch.long)
    mask[foreground] = class_id
    return composed, mask
