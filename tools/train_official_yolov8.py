from __future__ import annotations

import argparse
from pathlib import Path


def train(weights: str | Path, data: str | Path, project: str | Path, *, seed: int = 42, epochs: int = 100, batch: int = 4):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install ultralytics==8.4.0 from requirements-dev.txt") from exc
    import ultralytics
    if ultralytics.__version__ != "8.4.0":
        raise RuntimeError(f"official baseline requires ultralytics 8.4.0, got {ultralytics.__version__}")
    model = YOLO(str(weights))
    return model.train(
        data=str(data), imgsz=640, epochs=epochs, batch=batch, workers=0,
        optimizer="AdamW", lr0=1e-3, lrf=0.01, weight_decay=5e-4, warmup_epochs=3,
        cos_lr=True, mosaic=0.0, mixup=0.0, perspective=0.0, degrees=0.0, translate=0.0,
        scale=0.0, fliplr=0.0, seed=seed, deterministic=True, pretrained=True,
        project=str(project), name=f"yolov8n-seed{seed}", exist_ok=False, val=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pinned official YOLOv8 v8.4.0 baseline")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    print(train(args.weights, args.data, args.project, seed=args.seed, epochs=args.epochs, batch=args.batch))


if __name__ == "__main__":
    main()
