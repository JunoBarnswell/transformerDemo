from __future__ import annotations

import torch

from chatdemo.vision.metrics import evaluate_detection, evaluate_segmentation
from chatdemo.vision.types import Detection


def test_detection_metrics_use_detector_boxes_and_canonical_targets():
    result = evaluate_detection(
        [[Detection(0, 0.95, (10.0, 10.0, 50.0, 50.0))]],
        [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])}],
        num_classes=1,
    )
    assert result["map50"] == 1.0
    assert result["map75"] == 1.0


def test_segmentation_metrics_report_per_class_values():
    mask = torch.zeros((4, 4), dtype=torch.long)
    mask[1:3, 1:3] = 1
    result = evaluate_segmentation([mask], [mask], num_classes=2)
    assert result["miou"] == 1.0
    assert result["dice"] == 1.0
