from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader

from .config import VisionConfig
from .data import VisionDataset, collate_vision_batch
from .losses import DetectionLoss
from .model import DefectVisionModel, initialize_vision_model, load_vision_checkpoint, save_vision_checkpoint
from .fewshot import select_fewshot_samples


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = current[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


def _resolve(value: str | Path, config_path: Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    return config_path, payload


def _scheduler_factor(epoch: int, max_epochs: int, warmup: int, minimum: float) -> float:
    if epoch < warmup:
        return 0.1 + 0.9 * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(max_epochs - warmup - 1, 1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def train_from_config(config_path: str | Path, *, stage_override: str | None = None) -> Path:
    config_file, payload = _load(config_path)
    config = VisionConfig.from_dict(payload)
    data = payload.get("data", {})
    train = payload.get("training", {})
    init = payload.get("vision_init", {})
    if not all(isinstance(item, dict) for item in (data, train, init)):
        raise ValueError("data, training and vision_init must be mappings")
    manifest = _resolve(data.get("manifest", ""), config_file)
    if not manifest.is_file():
        raise FileNotFoundError(f"vision manifest not found: {manifest}")
    seed = int(payload.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = VisionDataset(manifest, input_size=640, num_classes=config.num_classes)
    if any(not sample.has_detection_labels for sample in dataset.samples):
        dataset.samples = [sample for sample in dataset.samples if sample.has_detection_labels]
    if not dataset.samples:
        raise ValueError("detection training requires detection-labeled samples")
    validation = None
    if data.get("validation_manifest"):
        validation_manifest = _resolve(data["validation_manifest"], config_file)
        validation = VisionDataset(validation_manifest, input_size=640, num_classes=config.num_classes)
    mode = str(init.get("mode", "random")).lower()
    if train.get("fewshot_shots") is not None:
        if mode == "random":
            raise ValueError("few-shot training requires a real vision_base checkpoint")
        dataset.samples = select_fewshot_samples(dataset.samples, images_per_class=int(train["fewshot_shots"]), seed=seed)
    checkpoint = _resolve(init["checkpoint"], config_file) if init.get("checkpoint") else None
    mapping = _resolve(init["mapping"], config_file) if init.get("mapping") else None
    stage = (stage_override or train.get("stage", "baseline")).lower()
    if mode == "random" and stage != "baseline":
        raise ValueError("random initialization is only valid for the fully unfrozen baseline stage")
    resume_path = _resolve(train["resume_from"], config_file) if train.get("resume_from") else None
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        model, resumed_config, resume_payload = load_vision_checkpoint(resume_path)
        if resumed_config.to_dict() != config.to_dict():
            raise ValueError("resume checkpoint config does not exactly match requested config")
        metadata = resume_payload.get("extra", {})
        if isinstance(metadata, dict) and metadata.get("manifest_sha256") not in (None, _fingerprint(manifest)):
            raise ValueError("resume checkpoint manifest fingerprint does not match")
    else:
        model = initialize_vision_model(config, mode=mode, checkpoint=checkpoint, mapping=mapping)
    model.freeze_for_stage(stage)
    device = torch.device(str(train.get("device", "cpu")))
    model.to(device)
    batch_size = int(train.get("batch_size", 4))
    accumulation = int(train.get("gradient_accumulation", 16))
    max_epochs = int(train.get("max_epochs", 100))
    if min(batch_size, accumulation, max_epochs) <= 0:
        raise ValueError("batch_size, gradient_accumulation and max_epochs must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(train.get("num_workers", 0)), collate_fn=collate_vision_batch, generator=torch.Generator().manual_seed(seed))
    base_lr = float(train.get("learning_rate", 1e-3))
    box_lr = float(train.get("box_learning_rate", base_lr * 0.1))
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "initial_lr": base_lr}]
    box_ids = {id(p) for module in (model.detection_head.cv2,) for p in module.parameters() if p.requires_grad}
    groups[0]["params"] = [p for p in groups[0]["params"] if id(p) not in box_ids]
    groups.append({"params": [p for p in model.parameters() if p.requires_grad and id(p) in box_ids], "lr": box_lr, "initial_lr": box_lr})
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train.get("weight_decay", 5e-4)))
    scheduler_warmup = int(train.get("warmup_epochs", 3))
    minimum = float(train.get("min_lr_factor", 0.01))
    criterion = DetectionLoss(config.num_classes)
    ema = ModelEMA(model, decay=float(train.get("ema_decay", 0.9999)))
    start_epoch = 0
    if resume_payload is not None:
        if isinstance(resume_payload.get("optimizer_state"), dict):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if isinstance(resume_payload.get("ema_state"), dict):
            ema.module.load_state_dict(resume_payload["ema_state"], strict=True)
        start_epoch = int(resume_payload.get("extra", {}).get("epoch", 0)) if isinstance(resume_payload.get("extra"), dict) else 0
    output = _resolve(train.get("output_dir", "outputs/vision"), config_file)
    if resume_path is None and any((output / name).exists() for name in ("run_metadata.json", "best.pt", "last.pt")):
        raise FileExistsError(f"output directory already contains a run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({"format": "chatdemo-detection-v3/v4", "manifest": str(manifest), "manifest_sha256": _fingerprint(manifest), "seed": seed, "stage": stage, "architecture": config.architecture, "initialization": mode}, indent=2) + "\n", encoding="utf-8")
    best_rank: tuple[float, float, float] | None = None
    if resume_payload is not None and isinstance(resume_payload.get("extra"), dict):
        previous = resume_payload["extra"].get("validation")
        if isinstance(previous, dict):
            best_rank = (float(previous.get("map50_95", -1.0)), float(previous.get("map75", -1.0)), float(previous.get("small_recall", -1.0)))
    patience = int(train.get("early_stopping_patience", 20))
    stale = 0
    log_file = (output / "train_log.jsonl").open("a" if resume_path is not None else "w", encoding="utf-8")
    try:
        for epoch in range(start_epoch, max_epochs):
            factor = _scheduler_factor(epoch, max_epochs, scheduler_warmup, minimum)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for step, (images, targets) in enumerate(loader):
                images = images.to(device)
                outputs = model(images, tasks=("detection",))
                result = criterion(outputs["detection"], [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
                (result["loss"] / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train.get("grad_clip", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                loss_sum += float(result["loss"].detach())
            record: dict[str, Any] = {"epoch": epoch + 1, "train_loss": loss_sum / len(loader), "learning_rate": [group["lr"] for group in optimizer.param_groups]}
            improved = False
            if validation is not None and (epoch + 1) % int(train.get("validation_interval", 1)) == 0:
                from .benchmark import evaluate_candidate
                val = evaluate_candidate_from_model(ema.module, config, validation, device)
                record["validation"] = val
                rank = (float(val["map50_95"]), float(val["map75"]), float(val["small_recall"]))
                improved = best_rank is None or rank > best_rank
                if improved:
                    best_rank, stale = rank, 0
                    save_vision_checkpoint(output / "best.pt", ema.module, extra={"epoch": epoch + 1, "validation": val, "selection": "map50_95_ap75_small_recall", "manifest_sha256": _fingerprint(manifest)})
                else:
                    stale += 1
            elif validation is None and (best_rank is None or record["train_loss"] < float(best_rank[0])):
                best_rank = (record["train_loss"], 0.0, 0.0)
                save_vision_checkpoint(output / "best.pt", ema.module, extra={"epoch": epoch + 1, "selection": "epoch_train_loss", "manifest_sha256": _fingerprint(manifest)})
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            save_vision_checkpoint(output / "last.pt", model, optimizer=optimizer, ema_state=ema.module.state_dict(), extra={"epoch": epoch + 1, "validation": record.get("validation"), "selection": "validation_map" if validation else "epoch_loss", "manifest_sha256": _fingerprint(manifest)})
            if validation is not None and patience > 0 and stale >= patience and epoch + 1 >= int(train.get("min_epochs", 30)):
                break
    finally:
        log_file.close()
    return output / "last.pt"


@torch.inference_mode()
def evaluate_candidate_from_model(model: DefectVisionModel, config: VisionConfig, dataset: VisionDataset, device: torch.device) -> dict[str, float]:
    from .coordinates import model_to_canonical
    from .postprocess import decode_detections
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_vision_batch)
    predictions, targets = [], []
    for images, batch in loader:
        output = model(images.to(device), tasks=("detection",))
        predictions.extend(decode_detections(output["detection"], [batch[0]["transform"]], confidence_threshold=0.001, nms_threshold=config.nms_threshold, max_detections=300, class_names=config.class_names))
        targets.append({"boxes": model_to_canonical(batch[0]["boxes"], batch[0]["transform"]), "labels": batch[0]["labels"]})
    from .metrics import evaluate_detection
    result = evaluate_detection(predictions, targets, num_classes=config.num_classes)
    result["small_recall"] = result.get("recall", 0.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train native YOLOv8 v8.4.0 detection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default=None)
    args = parser.parse_args()
    print(train_from_config(args.config, stage_override=args.stage))


if __name__ == "__main__":
    main()
