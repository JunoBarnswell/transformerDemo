from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from chatdemo.vision.model import load_vision_checkpoint
from chatdemo.vision.train import train_from_config


def test_detection_epoch_smoke_writes_v3_checkpoint(tmp_path: Path):
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), (80, 80, 80)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image": image.name, "boxes": [[10, 10, 40, 40]], "labels": [0]}) + "\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("\n".join([
        "vision:", "  reference: ultralytics-v8.4.0", "  architecture: yolov8n", "  input_size: 640",
        "  num_classes: 1", "  class_names: [defect]", "  confidence_threshold: 0.25", "  nms_threshold: 0.7", "  max_detections: 300",
        "vision_init:", "  mode: random", "data:", f"  manifest: {manifest.as_posix()}", "training:",
        "  stage: baseline", f"  output_dir: {(tmp_path / 'out').as_posix()}", "  batch_size: 1", "  gradient_accumulation: 1",
        "  max_epochs: 1", "  warmup_epochs: 0", "  device: cpu",
    ]) + "\n", encoding="utf-8")
    checkpoint = train_from_config(config)
    model, loaded, payload = load_vision_checkpoint(checkpoint)
    assert checkpoint.name == "last.pt"
    assert (checkpoint.parent / "best.pt").is_file()
    assert loaded.architecture == "yolov8n"
    assert payload["format"] == "chatdemo-detection-v3"
    assert model.training is False
