from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from .config import VisionConfig
from .data import VisionDataset, collate_vision_batch, estimate_max_object_area_fraction, filter_labeled_targets
from .fewshot import apply_fewshot_stage, parameter_groups, select_fewshot_samples
from .losses import DetectionLoss, SegmentationLoss
from .model import DefectVisionModel, initialize_vision_model, save_vision_checkpoint


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_file() or candidate.is_dir():
        return candidate
    from_config = config_path.parent / candidate
    if from_config.exists():
        return from_config
    return candidate


def _slice_detection_outputs(outputs: Mapping[str, Mapping[str, torch.Tensor]], indices: Sequence[int]) -> dict:
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=next(iter(outputs.values()))["box_logits"].device)
    return {
        level: {
            key: value.index_select(0, index_tensor)
            for key, value in level_outputs.items()
        }
        for level, level_outputs in outputs.items()
    }


def _select_targets(targets: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> list[Mapping[str, Any]]:
    return [targets[index] for index in indices]


def _load_payload(config_path: str | Path) -> tuple[Path, Dict[str, Any]]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    return path, payload


def train_from_config(
    config_path: str | Path,
    *,
    task_override: str | None = None,
    fewshot_stage_override: str | None = None,
) -> Path:
    config_file, payload = _load_payload(config_path)
    vision_config = VisionConfig.from_dict(payload.get("vision", payload))
    data_payload = payload.get("data", {})
    train_payload = payload.get("training", payload.get("train", {}))
    init_payload = payload.get("vision_init", {})
    if not isinstance(data_payload, dict) or not isinstance(train_payload, dict) or not isinstance(init_payload, dict):
        raise ValueError("data, training, and vision_init sections must be mappings")
    manifest = _resolve_path(data_payload.get("manifest", ""), config_file)
    if not manifest.is_file():
        raise FileNotFoundError(f"vision manifest not found: {manifest}")
    task = (task_override or train_payload.get("task", "multitask")).strip().lower()
    if task not in {"detection", "segmentation", "multitask"}:
        raise ValueError("training task must be detection, segmentation, or multitask")
    mode = str(init_payload.get("mode", "project_base"))
    checkpoint_value = init_payload.get("checkpoint")
    checkpoint = _resolve_path(checkpoint_value, config_file) if checkpoint_value else None
    mapping_value = init_payload.get("mapping")
    mapping = _resolve_path(mapping_value, config_file) if mapping_value else None
    model = initialize_vision_model(vision_config, mode=mode, checkpoint=checkpoint, mapping=mapping)
    stage = str(fewshot_stage_override or train_payload.get("fewshot_stage", "f4" if task == "multitask" else "f1"))
    apply_fewshot_stage(model, stage)

    dataset = VisionDataset(
        manifest,
        input_size=vision_config.input_size,
        num_classes=vision_config.num_classes,
        segmentation_classes=vision_config.segmentation_classes,
    )
    if train_payload.get("fewshot_shots") is not None:
        dataset.samples = select_fewshot_samples(
            dataset.samples,
            shots_per_class=int(train_payload["fewshot_shots"]),
            seed=int(payload.get("seed", 42)),
            include_normal_samples=bool(train_payload.get("include_normal_samples", True)),
        )
    batch_size = int(train_payload.get("batch_size", 2))
    if batch_size <= 0:
        raise ValueError("training batch_size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_payload.get("num_workers", 0)),
        collate_fn=collate_vision_batch,
    )
    learning_rate = float(train_payload.get("learning_rate", 1e-3))
    regression_learning_rate = float(train_payload.get("regression_learning_rate", learning_rate * 0.1))
    optimizer = torch.optim.AdamW(
        parameter_groups(model, learning_rate=learning_rate, regression_learning_rate=regression_learning_rate),
        weight_decay=float(train_payload.get("weight_decay", 1e-4)),
    )
    detection_loss = DetectionLoss(vision_config.num_classes, reg_max=vision_config.reg_max)
    segmentation_loss = SegmentationLoss(vision_config.segmentation_classes)
    device = torch.device(str(train_payload.get("device", "cpu")))
    model.to(device)
    max_steps = int(train_payload.get("max_steps", 100))
    if max_steps <= 0:
        raise ValueError("training max_steps must be positive")
    save_interval = int(train_payload.get("save_interval", 50))
    if save_interval <= 0:
        raise ValueError("training save_interval must be positive")
    output_dir = _resolve_path(train_payload.get("output_dir", "outputs/vision"), config_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / str(train_payload.get("log_file", "train_log.jsonl"))
    seed = int(payload.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    model.train()
    step = 0
    best_loss = float("inf")
    iterator = iter(loader)
    while step < max_steps:
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device)
        outputs = model(images, tasks=("detection", "segmentation") if task == "multitask" else (task,))
        total = images.sum() * 0.0
        details: Dict[str, float] = {}

        if task in {"detection", "multitask"}:
            detection_indices, _ = filter_labeled_targets(targets, "detection")
            if detection_indices:
                d_targets = _select_targets(targets, detection_indices)
                d_outputs = _slice_detection_outputs(outputs["detection"], detection_indices)
                d_result = detection_loss(d_outputs, [
                    {"boxes": item["boxes"], "labels": item["labels"]} for item in d_targets
                ], levels=vision_config.detect_levels)
                total = total + d_result["loss"]
                details.update({key: float(value.detach().item()) for key, value in d_result.items() if key != "loss"})

        if task in {"segmentation", "multitask"}:
            segmentation_indices, _ = filter_labeled_targets(targets, "segmentation")
            if segmentation_indices:
                masks = [targets[index]["mask"] for index in segmentation_indices]
                if any(mask is None for mask in masks):
                    raise ValueError("segmentation_labeled target is missing its mask tensor")
                mask_batch = torch.stack([mask for mask in masks if mask is not None], dim=0).to(device)
                index_tensor = torch.as_tensor(segmentation_indices, dtype=torch.long, device=device)
                s_result = segmentation_loss(outputs["segmentation"].index_select(0, index_tensor), mask_batch)
                total = total + s_result["loss"]
                details.update({key: float(value.detach().item()) for key, value in s_result.items() if key != "loss"})

        if not total.requires_grad:
            raise ValueError("batch contains no labels for the selected training task")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_payload.get("grad_clip", 1.0)))
        optimizer.step()
        step += 1
        loss_value = float(total.detach().item())
        details["loss"] = loss_value
        details["step"] = step
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(details, ensure_ascii=False) + "\n")
        if loss_value < best_loss:
            best_loss = loss_value
            save_vision_checkpoint(output_dir / "best.pt", model, extra={"task": task, "step": step, "best_loss": best_loss})
        if step % save_interval == 0 or step == max_steps:
            save_vision_checkpoint(output_dir / "checkpoint.pt", model, extra={"task": task, "step": step, "best_loss": best_loss})
    return output_dir / "checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the shared industrial-vision model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=("detection", "segmentation", "multitask"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(train_from_config(args.config, task_override=args.task))


if __name__ == "__main__":
    main()
