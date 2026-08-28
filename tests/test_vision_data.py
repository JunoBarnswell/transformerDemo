from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from chatdemo.vision.data import VisionDataset, load_manifest
from chatdemo.vision.fewshot import select_fewshot_samples


def _write_image(path: Path, color: tuple[int, int, int] = (20, 30, 40)) -> None:
    Image.new("RGB", (80, 40), color=color).save(path)


def test_manifest_preserves_missing_and_explicit_negative_labels(tmp_path: Path):
    normal = tmp_path / "normal.png"
    box = tmp_path / "box.png"
    mask_image = tmp_path / "mask.png"
    mask = tmp_path / "mask_label.png"
    _write_image(normal)
    _write_image(box)
    _write_image(mask_image)
    Image.new("L", (80, 40), color=0).save(mask)
    payload = [
        {"image": normal.name, "boxes": [], "labels": []},
        {"image": box.name, "boxes": [[5, 5, 30, 20]], "labels": [0]},
        {"image": mask_image.name, "mask": mask.name},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in payload) + "\n", encoding="utf-8")

    samples = load_manifest(manifest)
    assert samples[0].has_detection_labels and not samples[0].has_segmentation_label
    assert samples[1].has_detection_labels
    assert not samples[2].has_detection_labels and samples[2].has_segmentation_label

    dataset = VisionDataset(manifest, input_size=640, num_classes=1)
    _, negative = dataset[0]
    _, segmentation_only = dataset[2]
    assert negative["detection_labeled"] is True
    assert negative["boxes"].shape == (0, 4)
    assert segmentation_only["detection_labeled"] is False
    assert segmentation_only["segmentation_labeled"] is True


def test_fewshot_selection_requires_real_instances(tmp_path: Path):
    image = tmp_path / "image.png"
    _write_image(image)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image": image.name, "boxes": [[1, 1, 10, 10]], "labels": [0]}) + "\n",
        encoding="utf-8",
    )
    samples = load_manifest(manifest)
    selected = select_fewshot_samples(samples, shots_per_class=1)
    assert len(selected) == 1
