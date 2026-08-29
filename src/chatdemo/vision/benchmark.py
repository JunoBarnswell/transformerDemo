from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .coordinates import model_to_canonical
from .data import VisionDataset, collate_vision_batch
from .metrics import calibrate_detection_threshold, evaluate_detection
from .model import load_vision_checkpoint
from .postprocess import decode_detections


@torch.inference_mode()
def evaluate_candidate(checkpoint: str | Path, manifest: str | Path, *, batch_size: int = 1, confidence_threshold: float = 0.001) -> dict[str, Any]:
    if confidence_threshold != 0.001:
        raise ValueError("AP evaluation is fixed at confidence_threshold=0.001")
    model, config, _ = load_vision_checkpoint(checkpoint)
    dataset = VisionDataset(manifest, input_size=config.input_size, num_classes=config.num_classes)
    device = next(model.parameters()).device
    predictions, targets = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_vision_batch)
    for images, batch_targets in loader:
        outputs = model(images.to(device), tasks=("detection",))
        decoded = decode_detections(outputs["detection"], [target["transform"] for target in batch_targets], confidence_threshold=0.001, nms_threshold=config.nms_threshold, max_detections=300, class_names=config.class_names)
        for prediction, target in zip(decoded, batch_targets):
            if not target["detection_labeled"]:
                continue
            predictions.append(prediction)
            targets.append({"boxes": model_to_canonical(target["boxes"], target["transform"]), "labels": target["labels"]})
    if not targets:
        raise ValueError("benchmark manifest contains no detection-labeled samples")
    return {"samples": len(targets), "candidate": evaluate_detection(predictions, targets, num_classes=config.num_classes), "thresholds": calibrate_detection_threshold(predictions, targets, num_classes=config.num_classes)}


def run_benchmark(checkpoint: str | Path, manifest: str | Path, yolov8_weights: str | Path | None = None) -> dict[str, Any]:
    if yolov8_weights is None or not Path(yolov8_weights).is_file():
        raise FileNotFoundError("an explicit v8.4.0 yolov8n.pt is required for the official baseline")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install the pinned ultralytics==8.4.0 development dependency") from exc
    result = evaluate_candidate(checkpoint, manifest)
    model, config, _ = load_vision_checkpoint(checkpoint)
    del model
    dataset = VisionDataset(manifest, input_size=config.input_size, num_classes=config.num_classes)
    official = YOLO(str(yolov8_weights))
    predictions, targets = [], []
    from PIL import Image
    from .types import Detection
    from .coordinates import original_to_canonical
    for sample in dataset.samples:
        if not sample.has_detection_labels:
            continue
        with Image.open(sample.image_path) as image:
            width, height = image.width, image.height
        output = official.predict(source=sample.image_path, imgsz=640, conf=0.001, max_det=300, iou=config.nms_threshold, verbose=False)[0]
        current = []
        if output.boxes is not None:
            boxes = original_to_canonical(output.boxes.xyxy.detach().cpu().float(), width, height)
            for box, score, label in zip(boxes.tolist(), output.boxes.conf.detach().cpu().tolist(), output.boxes.cls.detach().cpu().tolist()):
                current.append(Detection(int(label), float(score), tuple(box)))
        predictions.append(current)
        targets.append({"boxes": original_to_canonical(torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4), width, height), "labels": torch.tensor(sample.labels, dtype=torch.long)})
    result["yolov8"] = evaluate_detection(predictions, targets, num_classes=config.num_classes)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v8.4.0 detector benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolov8-weights", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_benchmark(args.checkpoint, args.manifest, args.yolov8_weights)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
