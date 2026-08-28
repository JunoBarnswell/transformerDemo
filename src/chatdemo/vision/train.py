from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from .augment import ConservativeAugment
from .benchmark import evaluate_model
from .config import VisionConfig
from .data import VisionDataset, collate_vision_batch, estimate_max_object_area_fraction, filter_labeled_targets
from .fewshot import apply_fewshot_stage, parameter_groups, select_fewshot_samples
from .losses import DetectionLoss, SegmentationLoss
from .model import initialize_vision_model, save_vision_checkpoint


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    """Resolve every relative path against the owning config file."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _slice_detection_outputs(outputs: Mapping[str, Mapping[str, torch.Tensor]], indices: Sequence[int]) -> dict:
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=next(iter(outputs.values()))["box_logits"].device)
    return {
        level: {key: value.index_select(0, index_tensor) for key, value in level_outputs.items()}
        for level, level_outputs in outputs.items()
    }


def _load_payload(config_path: str | Path) -> tuple[Path, Dict[str, Any]]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    return path, payload


def _build_augmentation(train_payload: Mapping[str, Any]) -> ConservativeAugment | None:
    payload = train_payload.get("augmentation")
    if payload is None or payload is False:
        return None
    if not isinstance(payload, dict):
        raise ValueError("training.augmentation must be a mapping or false")
    if not bool(payload.get("enabled", True)):
        return None
    return ConservativeAugment(
        horizontal_flip=float(payload.get("horizontal_flip", 0.0)),
        vertical_flip=float(payload.get("vertical_flip", 0.0)),
        rotate90=float(payload.get("rotate90", 0.0)),
        brightness=float(payload.get("brightness", 0.0)),
        contrast=float(payload.get("contrast", 0.0)),
    )


def _learning_rate_factor(epoch_index: int, max_epochs: int, warmup_epochs: int, min_factor: float) -> float:
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return 0.1 + 0.9 * float(epoch_index + 1) / float(warmup_epochs)
    remaining = max(max_epochs - warmup_epochs - 1, 1)
    progress = min(max(float(epoch_index - warmup_epochs) / float(remaining), 0.0), 1.0)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))


def _ensure_new_output_directory(path: Path) -> None:
    tracked_artifacts = ("run_metadata.json", "train_log.jsonl", "best.pt", "checkpoint.pt")
    if path.exists() and any((path / name).exists() for name in tracked_artifacts):
        raise FileExistsError(f"training output already contains a run; choose a new output_dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def train_from_config(
    config_path: str | Path,
    *,
    task_override: str | None = None,
    fewshot_stage_override: str | None = None,
) -> Path:
    config_file, payload = _load_payload(config_path)
    data_payload = payload.get("data", {})
    train_payload = payload.get("training", payload.get("train", {}))
    init_payload = payload.get("vision_init", {})
    if not isinstance(data_payload, dict) or not isinstance(train_payload, dict) or not isinstance(init_payload, dict):
        raise ValueError("data, training, and vision_init sections must be mappings")
    if "max_steps" in train_payload:
        raise ValueError("training.max_steps was removed; use epoch-complete training.max_epochs")

    seed = int(payload.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    requested_config = VisionConfig.from_dict(payload.get("vision", payload))
    manifest = _resolve_path(data_payload.get("manifest", ""), config_file)
    if not manifest.is_file():
        raise FileNotFoundError(f"vision manifest not found: {manifest}")
    task = str(task_override or train_payload.get("task", "multitask")).strip().lower()
    if task not in {"detection", "segmentation", "multitask"}:
        raise ValueError("training task must be detection, segmentation, or multitask")

    train_dataset = VisionDataset(
        manifest,
        input_size=requested_config.input_size,
        num_classes=requested_config.num_classes,
        segmentation_classes=requested_config.segmentation_classes,
        transform=_build_augmentation(train_payload),
    )
    if train_payload.get("fewshot_shots") is not None:
        train_dataset.samples = select_fewshot_samples(
            train_dataset.samples,
            shots_per_class=int(train_payload["fewshot_shots"]),
            seed=seed,
            include_normal_samples=bool(train_payload.get("include_normal_samples", True)),
        )
    max_area_fraction = estimate_max_object_area_fraction(train_dataset.samples)
    vision_config = requested_config.resolve_for_dataset(max_area_fraction)

    mode = str(init_payload.get("mode", "project_base")).strip().lower()
    checkpoint_value = init_payload.get("checkpoint")
    checkpoint = _resolve_path(checkpoint_value, config_file) if checkpoint_value else None
    mapping_value = init_payload.get("mapping")
    mapping = _resolve_path(mapping_value, config_file) if mapping_value else None
    stage = str(fewshot_stage_override or train_payload.get("fewshot_stage", "f4" if task == "multitask" else "f1"))
    if mode == "random" and stage.lower() not in {"f4", "joint"}:
        raise ValueError("random initialization requires f4/joint full unfreeze; frozen random features are invalid")
    model = initialize_vision_model(vision_config, mode=mode, checkpoint=checkpoint, mapping=mapping)
    apply_fewshot_stage(model, stage)

    validation_dataset = None
    validation_value = data_payload.get("validation_manifest")
    if validation_value:
        validation_manifest = _resolve_path(validation_value, config_file)
        if not validation_manifest.is_file():
            raise FileNotFoundError(f"validation manifest not found: {validation_manifest}")
        validation_dataset = VisionDataset(
            validation_manifest,
            input_size=vision_config.input_size,
            num_classes=vision_config.num_classes,
            segmentation_classes=vision_config.segmentation_classes,
        )

    batch_size = int(train_payload.get("batch_size", 2))
    if batch_size <= 0:
        raise ValueError("training batch_size must be positive")
    data_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_payload.get("num_workers", 0)),
        collate_fn=collate_vision_batch,
        generator=data_generator,
    )

    device = torch.device(str(train_payload.get("device", "cpu")))
    model.to(device)
    learning_rate = float(train_payload.get("learning_rate", 1e-3))
    regression_learning_rate = float(train_payload.get("regression_learning_rate", learning_rate))
    groups = parameter_groups(
        model,
        learning_rate=learning_rate,
        regression_learning_rate=regression_learning_rate,
    )
    for group in groups:
        group["initial_lr"] = group["lr"]
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(train_payload.get("weight_decay", 1e-4)),
    )
    detection_loss = DetectionLoss(vision_config.num_classes, reg_max=vision_config.reg_max)
    segmentation_loss = SegmentationLoss(vision_config.segmentation_classes)

    max_epochs = int(train_payload.get("max_epochs", 50))
    if max_epochs <= 0:
        raise ValueError("training max_epochs must be positive")
    warmup_epochs = int(train_payload.get("warmup_epochs", 3))
    if not 0 <= warmup_epochs < max_epochs:
        raise ValueError("warmup_epochs must be in [0, max_epochs)")
    min_lr_factor = float(train_payload.get("min_lr_factor", 0.01))
    if not 0.0 < min_lr_factor <= 1.0:
        raise ValueError("min_lr_factor must be in (0,1]")
    save_interval = int(train_payload.get("save_interval", 10))
    validation_interval = int(train_payload.get("validation_interval", 1))
    if save_interval <= 0 or validation_interval <= 0:
        raise ValueError("save_interval and validation_interval must be positive")
    patience = int(train_payload.get("early_stopping_patience", 0))
    min_epochs = int(train_payload.get("min_epochs", 1))
    evaluation_threshold = float(train_payload.get("evaluation_confidence_threshold", 0.001))

    output_dir = _resolve_path(train_payload.get("output_dir", "../outputs/vision"), config_file)
    _ensure_new_output_directory(output_dir)
    log_path = output_dir / "train_log.jsonl"
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_file": str(config_file),
                "train_manifest": str(manifest),
                "validation_manifest": str(validation_manifest) if validation_dataset is not None else None,
                "seed": seed,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
                "max_object_area_fraction": max_area_fraction,
                "resolved_vision_config": vision_config.to_dict(),
                "task": task,
                "initialization_mode": mode,
                "fewshot_stage": stage,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    best_rank: tuple[float, float] | None = None
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch_index in range(max_epochs):
            final_epoch = epoch_index + 1
            factor = _learning_rate_factor(epoch_index, max_epochs, warmup_epochs, min_lr_factor)
            for group in optimizer.param_groups:
                group["lr"] = float(group["initial_lr"]) * factor
            model.train()
            sums: Dict[str, float] = {}
            batch_count = 0

            for images, targets in loader:
                images = images.to(device)
                outputs = model(images, tasks=("detection", "segmentation") if task == "multitask" else (task,))
                total = images.sum() * 0.0
                batch_details: Dict[str, torch.Tensor] = {}

                if task in {"detection", "multitask"}:
                    detection_indices, _ = filter_labeled_targets(targets, "detection")
                    if detection_indices:
                        d_outputs = _slice_detection_outputs(outputs["detection"], detection_indices)
                        d_result = detection_loss(
                            d_outputs,
                            [
                                {"boxes": targets[index]["boxes"], "labels": targets[index]["labels"]}
                                for index in detection_indices
                            ],
                            levels=vision_config.detect_levels,
                        )
                        total = total + d_result["loss"]
                        batch_details.update({f"detection_{key}": value for key, value in d_result.items() if key != "loss"})

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
                        batch_details.update({f"segmentation_{key}": value for key, value in s_result.items() if key != "loss"})

                if not total.requires_grad:
                    raise ValueError("batch contains no labels for the selected training task")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_payload.get("grad_clip", 1.0)))
                optimizer.step()

                batch_count += 1
                sums["loss"] = sums.get("loss", 0.0) + float(total.detach().item())
                for key, value in batch_details.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().item())

            if batch_count == 0:
                raise ValueError("training dataset produced no batches")
            train_means = {key: value / batch_count for key, value in sums.items()}
            validation_result = None
            improved = False

            if validation_dataset is not None and (
                final_epoch % validation_interval == 0 or final_epoch == max_epochs
            ):
                validation_result = evaluate_model(
                    model,
                    vision_config,
                    validation_dataset,
                    batch_size=int(train_payload.get("validation_batch_size", 1)),
                    confidence_threshold=evaluation_threshold,
                )
                candidate = validation_result["candidate"]
                current_rank = (float(candidate["map50"]), float(candidate["map50_95"]))
                improved = best_rank is None or current_rank > best_rank
                if improved:
                    best_rank = current_rank
            elif validation_dataset is None:
                current_loss = train_means["loss"]
                improved = current_loss < best_train_loss
                if improved:
                    best_train_loss = current_loss

            record: Dict[str, Any] = {
                "epoch": final_epoch,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "train": train_means,
            }
            if validation_result is not None:
                record["validation"] = validation_result
            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_handle.flush()

            extra = {
                "task": task,
                "epoch": final_epoch,
                "train": train_means,
                "validation": validation_result,
                "selection": "validation_map" if validation_dataset is not None else "epoch_mean_loss",
            }
            if improved:
                save_vision_checkpoint(output_dir / "best.pt", model, extra=extra)
                epochs_without_improvement = 0
            elif validation_result is not None:
                epochs_without_improvement += 1
            if final_epoch % save_interval == 0:
                save_vision_checkpoint(output_dir / "checkpoint.pt", model, extra=extra)
            if (
                validation_dataset is not None
                and patience > 0
                and final_epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                break

    save_vision_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        extra={
            "task": task,
            "epoch": final_epoch,
            "selection": "validation_map" if validation_dataset is not None else "epoch_mean_loss",
        },
    )
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
