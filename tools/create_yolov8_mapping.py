from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.model import DefectVisionModel


def build(source_checkpoint: str | Path, output: str | Path, *, architecture: str = "yolov8n", num_classes: int = 6) -> Path:
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source = source_payload.get("model_state", source_payload.get("state_dict")) if isinstance(source_payload, dict) else None
    if source is None and isinstance(source_payload, dict) and hasattr(source_payload.get("model"), "state_dict"):
        source = source_payload["model"].state_dict()
    if source is None and isinstance(source_payload, dict):
        source = source_payload
    if not isinstance(source, dict):
        raise ValueError("source checkpoint must contain model_state/state_dict")
    config = VisionConfig(architecture=architecture, num_classes=num_classes, class_names=[f"class_{i}" for i in range(num_classes)])
    target = DefectVisionModel(config).state_dict()
    mappings = {}
    reinitialize = []
    for name, value in target.items():
        if ".cv3." in name and (name not in source or tuple(source[name].shape) != tuple(value.shape)):
            reinitialize.append(name)
        elif name in source and tuple(source[name].shape) == tuple(value.shape):
            mappings[name] = name
        else:
            raise ValueError(f"target tensor has no exact source counterpart: {name}")
    spec = {"reference": "ultralytics-v8.4.0", "source": str(Path(source_checkpoint).resolve()), "mappings": mappings, "reinitialize_targets": reinitialize}
    target_path = Path(output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an explicit YOLOv8 v8.4.0 state mapping")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", default="yolov8n", choices=("yolov8n", "yolov8n-p2"))
    parser.add_argument("--num-classes", type=int, default=6)
    args = parser.parse_args()
    print(build(args.source, args.output, architecture=args.architecture, num_classes=args.num_classes))


if __name__ == "__main__":
    main()
