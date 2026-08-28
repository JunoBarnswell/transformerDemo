from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .coordinates import original_to_model, validate_boxes
from .preprocess import _pil, letterbox_image, load_image
from .types import ImageTransform, VisionSample


def _resolve_path(value: Any, base_dir: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} file not found: {path}")
    return str(path)


def _parse_boxes(record: Mapping[str, Any]) -> tuple[list[list[float]], list[int], bool]:
    has_boxes = "boxes" in record or "labels" in record
    if not has_boxes:
        return [], [], False
    if "boxes" not in record:
        raise ValueError("labels cannot be supplied without boxes")
    raw_boxes = record["boxes"]
    if raw_boxes is None:
        raw_boxes = []
    if not isinstance(raw_boxes, list):
        raise ValueError("boxes must be a list of xyxy arrays or objects")
    raw_labels = record.get("labels")
    if raw_labels is not None and not isinstance(raw_labels, list):
        raise ValueError("labels must be a list")

    boxes: list[list[float]] = []
    labels: list[int] = []
    for index, item in enumerate(raw_boxes):
        if isinstance(item, dict):
            coordinates = item.get("box", item.get("box_xyxy"))
            label = item.get("label", item.get("class_id"))
            if label is None and raw_labels is not None and index < len(raw_labels):
                label = raw_labels[index]
        else:
            coordinates = item
            label = raw_labels[index] if raw_labels is not None and index < len(raw_labels) else None
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
            raise ValueError(f"boxes[{index}] must contain four xyxy coordinates")
        if label is None:
            raise ValueError(f"boxes[{index}] is missing its class label")
        try:
            parsed_box = [float(value) for value in coordinates]
            parsed_label = int(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid box or label at index {index}") from exc
        boxes.append(parsed_box)
        labels.append(parsed_label)
    if raw_labels is not None and len(raw_labels) != len(boxes):
        raise ValueError("labels length must equal boxes length")
    validate_boxes(boxes, name="manifest boxes")
    if any(label < 0 for label in labels):
        raise ValueError("manifest labels must be non-negative")
    return boxes, labels, True


def load_manifest(manifest_path: str | Path) -> list[VisionSample]:
    """Load and validate a mixed detection/segmentation JSONL manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"vision manifest not found: {path}")
    samples: list[VisionSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest row {line_number} must be an object")
            image_path = _resolve_path(record.get("image"), path.parent, f"image at row {line_number}")
            boxes, labels, detection_labeled = _parse_boxes(record)
            mask_value = record.get("mask")
            segmentation_labeled = mask_value is not None
            mask_path = None
            if segmentation_labeled:
                mask_path = _resolve_path(mask_value, path.parent, f"mask at row {line_number}")
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata at row {line_number} must be an object")
            samples.append(VisionSample(
                image_path=image_path,
                boxes=tuple(tuple(box) for box in boxes),
                labels=tuple(labels),
                mask_path=mask_path,
                has_detection_labels=detection_labeled,
                has_segmentation_label=segmentation_labeled,
                metadata=dict(metadata),
            ))
    if not samples:
        raise ValueError(f"vision manifest contains no samples: {path}")
    return samples


def _letterbox_mask(mask_image: object, transform: ImageTransform):
    Image = _pil()
    if mask_image.width != transform.orig_w or mask_image.height != transform.orig_h:
        raise ValueError("segmentation mask dimensions must exactly match the source image")
    resized = mask_image.resize((transform.resized_w, transform.resized_h), Image.Resampling.NEAREST)
    canvas = Image.new("L", (transform.input_size, transform.input_size), color=0)
    canvas.paste(resized, (round(transform.pad_x), round(transform.pad_y)))
    return torch.from_numpy(np.array(canvas, dtype=np.int64, copy=True))


class VisionDataset(Dataset):
    """Dataset that preserves missing-vs-negative labels for both tasks."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        input_size: int = 640,
        num_classes: int | None = None,
        segmentation_classes: int = 2,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]] = None,
    ):
        self.samples = load_manifest(manifest_path)
        self.input_size = int(input_size)
        self.num_classes = num_classes
        self.segmentation_classes = int(segmentation_classes)
        self.augment = transform
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.segmentation_classes < 2:
            raise ValueError("segmentation_classes must include background")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Dict[str, Any]]:
        sample = self.samples[index]
        image = load_image(sample.image_path)
        image_tensor, image_transform = letterbox_image(image, input_size=self.input_size)
        original_boxes = torch.as_tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
        if original_boxes.numel():
            if (original_boxes < 0).any() or (original_boxes[:, 0::2] > image.width).any() or (original_boxes[:, 1::2] > image.height).any():
                raise ValueError(f"box exceeds image dimensions: {sample.image_path}")
        model_boxes = original_to_model(original_boxes, image_transform)
        labels = torch.as_tensor(sample.labels, dtype=torch.long)
        if self.num_classes is not None and labels.numel() and (labels >= self.num_classes).any():
            raise ValueError(f"label outside num_classes for sample: {sample.image_path}")

        mask = None
        if sample.has_segmentation_label:
            if sample.mask_path is None:
                raise ValueError("segmentation label is marked present but mask_path is empty")
            mask_image = load_image(sample.mask_path).convert("L")
            mask = _letterbox_mask(mask_image, image_transform)
            if mask.numel() and int(mask.max()) >= self.segmentation_classes:
                raise ValueError(f"mask class outside segmentation_classes: {sample.mask_path}")

        if self.augment is not None:
            image_tensor, model_boxes, mask = self.augment(image_tensor, model_boxes, mask)

        target: Dict[str, Any] = {
            "boxes": model_boxes,
            "labels": labels,
            "detection_labeled": sample.has_detection_labels,
            "mask": mask,
            "segmentation_labeled": sample.has_segmentation_label,
            "image_path": sample.image_path,
            "transform": image_transform,
            "metadata": sample.metadata,
        }
        return image_tensor, target


def collate_vision_batch(batch: Sequence[tuple[torch.Tensor, Dict[str, Any]]]) -> tuple[torch.Tensor, list[Dict[str, Any]]]:
    if not batch:
        raise ValueError("cannot collate an empty vision batch")
    images = torch.stack([item[0] for item in batch], dim=0)
    return images, [item[1] for item in batch]


def filter_labeled_targets(targets: Sequence[Mapping[str, Any]], task: str) -> tuple[list[int], list[Mapping[str, Any]]]:
    if task not in {"detection", "segmentation"}:
        raise ValueError("task must be detection or segmentation")
    key = "detection_labeled" if task == "detection" else "segmentation_labeled"
    selected = [(index, target) for index, target in enumerate(targets) if bool(target.get(key, False))]
    return [index for index, _ in selected], [target for _, target in selected]


def estimate_max_object_area_fraction(samples: Sequence[VisionSample]) -> float | None:
    """Estimate the largest annotated box fraction for ``use_p5: auto``."""
    maximum = 0.0
    found = False
    for sample in samples:
        if not sample.has_detection_labels:
            continue
        image = load_image(sample.image_path)
        for box in sample.boxes:
            x1, y1, x2, y2 = (float(value) for value in box)
            maximum = max(maximum, ((x2 - x1) * (y2 - y1)) / float(image.width * image.height))
            found = True
    return maximum if found else None
