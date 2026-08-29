from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch.nn as nn

from .model import DefectVisionModel
from .types import VisionSample


def apply_fewshot_stage(model: DefectVisionModel, stage: str) -> DefectVisionModel:
    stage = stage.lower().strip()
    model.freeze_for_stage(stage)
    if stage != "baseline" and not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"few-shot stage {stage!r} leaves no trainable parameters")
    return model


def parameter_groups(model: DefectVisionModel, *, learning_rate: float, box_learning_rate: float | None = None) -> list[dict]:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    box_learning_rate = learning_rate * 0.1 if box_learning_rate is None else box_learning_rate
    box_ids = {id(parameter) for parameter in model.detection_head.cv2.parameters()}
    regular = [p for p in model.parameters() if p.requires_grad and id(p) not in box_ids]
    regression = [p for p in model.parameters() if p.requires_grad and id(p) in box_ids]
    groups = []
    if regular:
        groups.append({"params": regular, "lr": learning_rate})
    if regression:
        groups.append({"params": regression, "lr": box_learning_rate})
    if not groups:
        raise ValueError("no trainable parameters available")
    return groups


def select_fewshot_samples(samples: Sequence[VisionSample], *, images_per_class: int | None = None, shots_per_class: int | None = None, seed: int = 42) -> list[VisionSample]:
    """Select an exact number of unique source images per class."""
    if images_per_class is not None and shots_per_class is not None:
        raise ValueError("provide only images_per_class")
    images_per_class = images_per_class if images_per_class is not None else shots_per_class
    if images_per_class is None or images_per_class <= 0:
        raise ValueError("images_per_class must be positive")
    by_class: dict[int, list[VisionSample]] = {}
    normals: list[VisionSample] = []
    for sample in samples:
        if not sample.has_detection_labels or not sample.labels:
            normals.append(sample)
        for label in sorted(set(int(value) for value in sample.labels)):
            by_class.setdefault(label, []).append(sample)
    if not by_class:
        raise ValueError("few-shot selection requires detection labels")
    rng = random.Random(seed)
    selected: list[VisionSample] = []
    selected_paths: set[str] = set()
    for label in sorted(by_class):
        candidates = list(by_class[label])
        rng.shuffle(candidates)
        if len(candidates) < images_per_class:
            raise ValueError(f"class {label} has fewer than {images_per_class} unique images")
        for sample in candidates[:images_per_class]:
            if sample.image_path not in selected_paths:
                selected.append(sample)
                selected_paths.add(sample.image_path)
    selected.extend(sample for sample in normals if sample.image_path not in selected_paths)
    return selected
