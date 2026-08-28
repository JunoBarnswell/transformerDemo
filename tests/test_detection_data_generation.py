from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from chatdemo.vision.augment import transform_square_boxes
from tools.generate_balanced_detection_data import generate_balanced_dataset


def test_square_box_transform_matches_image_geometry():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    rotated = transform_square_boxes(boxes, size=100.0, quarter_turns=1)
    assert torch.equal(rotated, torch.tensor([[20.0, 70.0, 40.0, 90.0]]))
    mirrored = transform_square_boxes(rotated, size=100.0, mirror_horizontal=True)
    assert torch.equal(mirrored, torch.tensor([[60.0, 70.0, 80.0, 90.0]]))


def test_balanced_generator_produces_exact_per_class_counts(tmp_path: Path):
    rows = []
    for class_id in (0, 1):
        image = tmp_path / f"source_{class_id}.jpg"
        Image.new("RGB", (40, 40), color=(80 + class_id, 90, 100)).save(image)
        rows.append({"image": image.name, "boxes": [[5, 6, 20, 25]], "labels": [class_id]})
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    generated = generate_balanced_dataset(manifest, tmp_path / "generated", samples_per_class=3)
    output_rows = [json.loads(line) for line in generated.read_text(encoding="utf-8").splitlines()]
    assert len(output_rows) == 6
    assert [row["labels"][0] for row in output_rows].count(0) == 3
    assert [row["labels"][0] for row in output_rows].count(1) == 3
