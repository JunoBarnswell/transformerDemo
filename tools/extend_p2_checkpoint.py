from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.model import DefectVisionModel, save_vision_checkpoint


def convert(source_path: str | Path, output_path: str | Path) -> Path:
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source_payload, dict) or source_payload.get("architecture") != "yolov8n":
        raise ValueError("P2 conversion requires a v3 yolov8n checkpoint")
    source = source_payload.get("model_state")
    if not isinstance(source, dict):
        raise ValueError("source checkpoint is missing model_state")
    source_config = VisionConfig.from_dict(source_payload.get("vision_config", {}))
    target_config = VisionConfig(architecture="yolov8n-p2", num_classes=source_config.num_classes, class_names=list(source_config.class_names))
    target_model = DefectVisionModel(target_config)
    target = target_model.state_dict()
    mapping: dict[str, str] = {}
    for name in target:
        if name.startswith("model.28."):
            if ".cv3." in name or ".cv2.0." in name:
                continue
            source_name = name.replace("model.28.", "model.22.")
            for target_index, source_index in (("cv2.1.", "cv2.0."), ("cv2.2.", "cv2.1."), ("cv2.3.", "cv2.2."), ("cv3.1.", "cv3.0."), ("cv3.2.", "cv3.1."), ("cv3.3.", "cv3.2.")):
                source_name = source_name.replace(target_index, source_index)
            if source_name in source and tuple(source[source_name].shape) == tuple(target[name].shape):
                mapping[name] = source_name
            continue
        source_name = name
        if name.startswith("model.24."):
            source_name = name.replace("model.24.", "model.18.")
        elif name.startswith("model.25."):
            source_name = name.replace("model.25.", "model.19.")
        elif name.startswith("model.27."):
            source_name = name.replace("model.27.", "model.21.")
        elif name.startswith("model.") and int(name.split(".")[1]) >= 16:
            continue
        if source_name in source and tuple(source[source_name].shape) == tuple(target[name].shape):
            mapping[name] = source_name
    nontransfer = sum(1 for name in target if name.startswith(("model.18.", "model.19.", "model.21.", "model.22.", "model.28.cv2.0.", "model.28.cv3.")))
    if len(mapping) != len(target) - nontransfer:
        raise ValueError("P2 conversion did not cover every transferable tensor")
    converted = dict(target)
    for target_name, source_name in mapping.items():
        converted[target_name] = source[source_name].clone()
    target_model.load_state_dict(converted, strict=True)
    return save_vision_checkpoint(output_path, target_model, extra={"conversion": "explicit-yolov8n-to-yolov8n-p2", "source": str(Path(source_path).resolve()), "reinitialized": "P2 branch"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an accepted yolov8n checkpoint to the explicit P2 graph")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(convert(args.source, args.output))


if __name__ == "__main__":
    main()
