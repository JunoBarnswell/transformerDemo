from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.data import VisionDataset, collate_vision_batch
from chatdemo.vision.losses import DetectionLoss
from chatdemo.vision.metrics import evaluate_detection
from chatdemo.vision.model import DefectVisionModel
from chatdemo.vision.postprocess import decode_detections
from chatdemo.vision.coordinates import model_to_canonical


def run(manifest: str | Path, *, image_count: int, steps: int, device: str = "cpu", seed: int = 42) -> dict[str, float]:
    if image_count not in (1, 10) or steps <= 0:
        raise ValueError("image_count must be 1 or 10 and steps must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = VisionDataset(manifest, input_size=640, num_classes=None)
    class_count = max((max((int(label) for label in sample.labels), default=-1) for sample in dataset.samples), default=0) + 1
    indices = list(range(min(image_count, len(dataset))))
    dataset = Subset(dataset, indices)
    config = VisionConfig(num_classes=class_count, class_names=[f"class_{index}" for index in range(class_count)])
    model = DefectVisionModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = DetectionLoss(class_count)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_vision_batch)
    batches = list(loader)
    for step in range(steps):
        images, targets = batches[step % len(batches)]
        output = model(images.to(device), tasks=("detection",))["detection"]
        loss = criterion(output, [{"boxes": targets[0]["boxes"], "labels": targets[0]["labels"]}])["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    predictions, expected = [], []
    model.eval()
    for images, targets in batches:
        output = model(images.to(device), tasks=("detection",))["detection"]
        predictions.extend(decode_detections(output, [targets[0]["transform"]], confidence_threshold=0.001, nms_threshold=0.7, max_detections=300, class_names=config.class_names))
        expected.append({"boxes": model_to_canonical(targets[0]["boxes"], targets[0]["transform"]), "labels": targets[0]["labels"]})
    result = evaluate_detection(predictions, expected, num_classes=class_count)
    return {"images": float(image_count), "steps": float(steps), "map50": float(result["map50"]), "recall": float(result["recall"]), "small_recall": float(result["small_recall"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mandatory one-image or ten-image detection overfit gate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images", type=int, choices=(1, 10), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, image_count=args.images, steps=args.steps, device=args.device), indent=2))


if __name__ == "__main__":
    main()
