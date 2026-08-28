from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from .data import VisionDataset, collate_vision_batch
from .metrics import evaluate_detection, evaluate_segmentation
from .model import load_vision_checkpoint
from .postprocess import decode_detections
from .preprocess import restore_mask


def _canonical_targets(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, torch.Tensor]]:
    from .coordinates import model_to_canonical
    result = []
    for target in targets:
        transform = target["transform"]
        result.append({
            "boxes": model_to_canonical(target["boxes"], transform),
            "labels": target["labels"],
        })
    return result


@torch.inference_mode()
def evaluate_candidate(checkpoint: str | Path, manifest: str | Path, *, batch_size: int = 1) -> dict[str, Any]:
    model, config, _ = load_vision_checkpoint(checkpoint)
    dataset = VisionDataset(
        manifest,
        input_size=config.input_size,
        num_classes=config.num_classes,
        segmentation_classes=config.segmentation_classes,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_vision_batch)
    all_predictions = []
    all_targets = []
    segmentation_predictions = []
    segmentation_targets = []
    for images, targets in loader:
        outputs = model(images, tasks=("detection", "segmentation"))
        batch_predictions = decode_detections(
            outputs["detection"],
            [target["transform"] for target in targets],
            reg_max=config.reg_max,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold,
            max_detections=config.max_detections,
            class_names=config.class_names,
            levels=config.detect_levels,
        )
        batch_targets = _canonical_targets(targets)
        for prediction, target, metadata in zip(batch_predictions, batch_targets, targets):
            if metadata["detection_labeled"]:
                all_predictions.append(prediction)
                all_targets.append(target)
        masks = outputs["segmentation"].argmax(dim=1)
        for index, target in enumerate(targets):
            if not target["segmentation_labeled"]:
                continue
            restored = restore_mask(masks[index], target["transform"], mode="nearest").round().long()
            segmentation_predictions.append(restored)
            segmentation_targets.append(restore_mask(target["mask"], target["transform"], mode="nearest").long())
    result: dict[str, Any] = {
        "samples": len(dataset),
    }
    if not all_targets:
        raise ValueError("benchmark manifest contains no detection-labeled samples")
    result["candidate"] = evaluate_detection(all_predictions, all_targets, num_classes=config.num_classes)
    if segmentation_predictions:
        result["segmentation"] = evaluate_segmentation(
            segmentation_predictions, segmentation_targets, num_classes=config.segmentation_classes
        )
    return result


def run_benchmark(checkpoint: str | Path, manifest: str | Path, yolov8_weights: str | Path | None = None) -> dict[str, Any]:
    if yolov8_weights is None:
        raise ValueError("same-shot benchmark requires --yolov8-weights; candidate-only scores are not parity evidence")
    yolov8_path = Path(yolov8_weights)
    if not yolov8_path.is_file():
        raise FileNotFoundError(f"YOLOv8 baseline weights not found: {yolov8_path}")
    candidate = evaluate_candidate(checkpoint, manifest)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLOv8 baseline requires the separately reviewed ultralytics dependency") from exc
    model, config, _ = load_vision_checkpoint(checkpoint)
    del model
    dataset = VisionDataset(manifest, input_size=config.input_size, num_classes=config.num_classes, segmentation_classes=config.segmentation_classes)
    baseline_model = YOLO(str(yolov8_path))
    predictions = []
    targets = []
    from .coordinates import original_to_canonical
    from .types import Detection
    from PIL import Image
    for sample in dataset.samples:
        result = baseline_model.predict(source=sample.image_path, imgsz=config.input_size, verbose=False)[0]
        with Image.open(sample.image_path) as loaded:
            image_width, image_height = loaded.width, loaded.height
        sample_predictions = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().float()
            scores = result.boxes.conf.detach().cpu().float()
            labels = result.boxes.cls.detach().cpu().long()
            boxes = original_to_canonical(boxes, image_width, image_height, config.input_size)
            for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                sample_predictions.append(Detection(int(label), float(score), tuple(box)))
        if sample.has_detection_labels:
            predictions.append(sample_predictions)
            raw_boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
            targets.append({"boxes": original_to_canonical(raw_boxes, image_width, image_height, config.input_size), "labels": torch.tensor(sample.labels, dtype=torch.long)})
    return {"candidate": candidate, "yolov8": evaluate_detection(predictions, targets, num_classes=config.num_classes)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a same-shot detector benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolov8-weights", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args.checkpoint, args.manifest, args.yolov8_weights)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
