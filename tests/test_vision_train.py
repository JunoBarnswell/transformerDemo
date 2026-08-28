from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from chatdemo.vision.model import load_vision_checkpoint
from chatdemo.vision.train import train_from_config


def test_random_detection_training_writes_strict_checkpoint(tmp_path: Path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (96, 64), color=(90, 90, 90)).save(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "boxes": [[10, 10, 40, 30]], "labels": [0]}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "vision.yaml"
    config.write_text(
        "\n".join([
            "vision:",
            "  input_size: 640",
            "  num_classes: 1",
            "  class_names: [defect]",
            "  segmentation_classes: 2",
            "  backbone_channels: [4, 8, 16, 32, 64]",
            "  backbone_depths: [1, 1, 1, 1, 1]",
            "  neck_channels: [4, 8, 16, 32, 64]",
            "  use_p5: false",
            "  detect_levels: [P2, P3, P4]",
            "vision_init:",
            "  mode: random",
            "data:",
            f"  manifest: {manifest.as_posix()}",
            "training:",
            "  task: detection",
            "  fewshot_stage: f1",
            f"  output_dir: {(tmp_path / 'out').as_posix()}",
            "  batch_size: 1",
            "  max_steps: 1",
            "  save_interval: 1",
            "  device: cpu",
        ]) + "\n",
        encoding="utf-8",
    )
    checkpoint = train_from_config(config)
    assert checkpoint.is_file()
    model, loaded_config, _ = load_vision_checkpoint(checkpoint)
    assert loaded_config.num_classes == 1
    assert model.training is False
