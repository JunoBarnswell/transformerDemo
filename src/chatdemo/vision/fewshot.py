from __future__ import annotations

import random
from typing import Iterable, Sequence

import torch.nn as nn

from .model import DefectVisionModel
from .types import VisionSample


def apply_fewshot_stage(model: DefectVisionModel, stage: str) -> DefectVisionModel:
    """Apply F0-F4 ownership rules and return the same model for composition."""
    model.freeze_for_stage(stage)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"few-shot stage {stage!r} leaves no trainable parameters")
    return model


def parameter_groups(
    model: DefectVisionModel,
    *,
    learning_rate: float,
    regression_learning_rate: float | None = None,
) -> list[dict]:
    """Build optimizer groups without silently training frozen parameters."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    regression_learning_rate = regression_learning_rate or learning_rate
    default_parameters = []
    regression_parameters = []
    regression_ids = {
        id(parameter)
        for module in (model.detection_head.reg_towers, model.detection_head.reg_preds)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in regression_ids:
            regression_parameters.append(parameter)
        else:
            default_parameters.append(parameter)
    groups = []
    if default_parameters:
        groups.append({"params": default_parameters, "lr": learning_rate})
    if regression_parameters:
        groups.append({"params": regression_parameters, "lr": regression_learning_rate})
    if not groups:
        raise ValueError("no trainable parameters available")
    return groups


def select_fewshot_samples(
    samples: Sequence[VisionSample],
    *,
    shots_per_class: int,
    seed: int = 42,
    include_normal_samples: bool = True,
) -> list[VisionSample]:
    """Select exact per-class instance budgets without fabricating labels."""
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive")
    class_to_samples: dict[int, list[VisionSample]] = {}
    normal_samples: list[VisionSample] = []
    for sample in samples:
        if not sample.has_detection_labels or not sample.labels:
            normal_samples.append(sample)
        for label in set(int(value) for value in sample.labels):
            class_to_samples.setdefault(label, []).append(sample)
    if not class_to_samples:
        raise ValueError("few-shot selection requires at least one detection-labeled class")
    rng = random.Random(seed)
    selected: list[VisionSample] = []
    selected_ids: set[int] = set()
    for label in sorted(class_to_samples):
        candidates = list(class_to_samples[label])
        rng.shuffle(candidates)
        selected_for_class = 0
        for sample in candidates:
            if id(sample) not in selected_ids:
                selected.append(sample)
                selected_ids.add(id(sample))
            selected_for_class += sum(1 for item_label in sample.labels if int(item_label) == label)
            if selected_for_class >= shots_per_class:
                break
        if selected_for_class < shots_per_class:
            raise ValueError(f"class {label} has fewer than {shots_per_class} labeled instances")
    if include_normal_samples:
        selected.extend(sample for sample in normal_samples if id(sample) not in selected_ids)
    return selected
