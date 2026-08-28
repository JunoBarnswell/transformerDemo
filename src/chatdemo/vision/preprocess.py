from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from .types import ImageTransform


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("Pillow is required for image inference; install requirements.txt") from exc
    return Image


def load_image(image: Union[str, Path, object]):
    Image = _pil()
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        with Image.open(path) as loaded:
            return loaded.convert("RGB")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a filesystem path or PIL.Image.Image")
    return image.convert("RGB")


def build_image_transform(width: int, height: int, input_size: int = 640) -> ImageTransform:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    scale = min(float(input_size) / float(width), float(input_size) / float(height))
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    return ImageTransform(
        orig_w=width,
        orig_h=height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        input_size=input_size,
    )


def letterbox_image(image: object, input_size: int = 640, fill: int = 114) -> tuple[torch.Tensor, ImageTransform]:
    """Convert an image to normalized CHW tensor plus reversible geometry."""
    Image = _pil()
    rgb = load_image(image)
    transform = build_image_transform(rgb.width, rgb.height, input_size=input_size)
    resized = rgb.resize((transform.resized_w, transform.resized_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), color=(fill, fill, fill))
    canvas.paste((resized), (round(transform.pad_x), round(transform.pad_y)))
    array = np.array(canvas, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, transform


def restore_mask(mask: torch.Tensor, transform: ImageTransform, *, mode: str = "nearest") -> torch.Tensor:
    """Crop letterbox padding and resize a model-space mask to original size."""
    if mask.ndim not in (2, 3, 4):
        raise ValueError("mask must have shape [H,W], [C,H,W], or [B,C,H,W]")
    original_ndim = mask.ndim
    if mask.ndim == 2:
        working = mask[None, None]
    elif mask.ndim == 3:
        working = mask[None]
    else:
        working = mask
    left = round(transform.pad_x)
    top = round(transform.pad_y)
    right = left + transform.resized_w
    bottom = top + transform.resized_h
    if working.shape[-2] < bottom or working.shape[-1] < right:
        raise ValueError("mask spatial dimensions are smaller than the transform input size")
    cropped = working[..., top:bottom, left:right]
    if mode == "nearest":
        restored = F.interpolate(cropped.float(), size=(transform.orig_h, transform.orig_w), mode=mode)
    elif mode in {"bilinear", "bicubic"}:
        restored = F.interpolate(cropped.float(), size=(transform.orig_h, transform.orig_w), mode=mode, align_corners=False)
    else:
        raise ValueError(f"unsupported mask interpolation mode: {mode}")
    if original_ndim == 2:
        return restored[0, 0]
    if original_ndim == 3:
        return restored[0]
    return restored
