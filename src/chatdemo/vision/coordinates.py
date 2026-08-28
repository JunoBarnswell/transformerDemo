from __future__ import annotations

from typing import Sequence

import torch

from .types import ImageTransform


CANONICAL_SIZE = 640


def _box_tensor(boxes: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    tensor = boxes if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes)
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.float32).reshape(-1, 4)
    if tensor.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [..., 4], got {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor


def validate_boxes(boxes: torch.Tensor | Sequence[Sequence[float]], *, name: str = "boxes") -> torch.Tensor:
    tensor = _box_tensor(boxes)
    if tensor.numel() and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if tensor.numel() and (tensor[..., 2:] <= tensor[..., :2]).any():
        raise ValueError(f"{name} must use xyxy boxes with positive area")
    return tensor


def clamp_boxes(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    width: float,
    height: float,
) -> torch.Tensor:
    tensor = _box_tensor(boxes).clone()
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if tensor.numel() == 0:
        return tensor
    tensor[..., 0::2] = tensor[..., 0::2].clamp(0.0, float(width))
    tensor[..., 1::2] = tensor[..., 1::2].clamp(0.0, float(height))
    return tensor


def original_to_model(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    transform: ImageTransform,
) -> torch.Tensor:
    """Map original-image xyxy boxes into letterboxed model coordinates."""
    tensor = _box_tensor(boxes).clone()
    if tensor.numel() == 0:
        return tensor
    tensor[..., 0::2] = tensor[..., 0::2] * transform.scale + transform.pad_x
    tensor[..., 1::2] = tensor[..., 1::2] * transform.scale + transform.pad_y
    return clamp_boxes(tensor, transform.input_size, transform.input_size)


def model_to_original(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    transform: ImageTransform,
) -> torch.Tensor:
    """Map letterboxed model coordinates back to original-image coordinates."""
    tensor = _box_tensor(boxes).clone()
    if tensor.numel() == 0:
        return tensor
    tensor[..., 0::2] = (tensor[..., 0::2] - transform.pad_x) / transform.scale
    tensor[..., 1::2] = (tensor[..., 1::2] - transform.pad_y) / transform.scale
    return clamp_boxes(tensor, transform.orig_w, transform.orig_h)


def original_to_canonical(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    canonical_size: int = CANONICAL_SIZE,
) -> torch.Tensor:
    """Map original-image boxes into the public square coordinate system."""
    tensor = _box_tensor(boxes).clone()
    if orig_w <= 0 or orig_h <= 0 or canonical_size <= 0:
        raise ValueError("image and canonical dimensions must be positive")
    if tensor.numel() == 0:
        return tensor
    tensor[..., 0::2] *= float(canonical_size) / float(orig_w)
    tensor[..., 1::2] *= float(canonical_size) / float(orig_h)
    return clamp_boxes(tensor, canonical_size, canonical_size)


def canonical_to_original(
    boxes: torch.Tensor | Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    canonical_size: int = CANONICAL_SIZE,
) -> torch.Tensor:
    """Map public canonical 640 boxes back to original-image coordinates."""
    tensor = _box_tensor(boxes).clone()
    if orig_w <= 0 or orig_h <= 0 or canonical_size <= 0:
        raise ValueError("image and canonical dimensions must be positive")
    if tensor.numel() == 0:
        return tensor
    tensor[..., 0::2] *= float(orig_w) / float(canonical_size)
    tensor[..., 1::2] *= float(orig_h) / float(canonical_size)
    return clamp_boxes(tensor, orig_w, orig_h)


def model_to_canonical(boxes: torch.Tensor | Sequence[Sequence[float]], transform: ImageTransform) -> torch.Tensor:
    return original_to_canonical(
        model_to_original(boxes, transform),
        transform.orig_w,
        transform.orig_h,
        canonical_size=transform.input_size,
    )


def canonical_to_model(boxes: torch.Tensor | Sequence[Sequence[float]], transform: ImageTransform) -> torch.Tensor:
    return original_to_model(
        canonical_to_original(
            boxes,
            transform.orig_w,
            transform.orig_h,
            canonical_size=transform.input_size,
        ),
        transform,
    )


def box_area(boxes: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    tensor = _box_tensor(boxes)
    if tensor.numel() == 0:
        return tensor.new_zeros(tensor.shape[:-1])
    return (tensor[..., 2] - tensor[..., 0]).clamp_min(0) * (tensor[..., 3] - tensor[..., 1]).clamp_min(0)


def box_iou(boxes1: torch.Tensor | Sequence[Sequence[float]], boxes2: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    first = _box_tensor(boxes1)
    second = _box_tensor(boxes2)
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    lt = torch.maximum(first[:, None, :2], second[None, :, :2])
    rb = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return intersection / union.clamp_min(torch.finfo(first.dtype).eps)
