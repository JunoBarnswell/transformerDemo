from __future__ import annotations

import torch

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.losses import DetectionLoss, SegmentationLoss
from chatdemo.vision.model import DefectVisionModel, initialize_vision_model


def _config() -> VisionConfig:
    return VisionConfig(
        num_classes=2,
        class_names=["scratch", "pit"],
        segmentation_classes=3,
        backbone_channels=(4, 8, 16, 32, 64),
        backbone_depths=(1, 1, 1, 1, 1),
        neck_channels=(4, 8, 16, 32, 64),
        use_p5=False,
        detect_levels=("P2", "P3", "P4"),
    )


def test_shared_trunk_has_independent_detection_and_segmentation_outputs():
    model = DefectVisionModel(_config()).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 640, 640))
    assert set(output) == {"detection", "segmentation"}
    assert set(output["detection"]) == {"P2", "P3", "P4"}
    assert output["detection"]["P2"]["box_logits"].shape[1] == 4 * 17
    assert output["detection"]["P2"]["cls_logits"].shape[1] == 2
    assert output["segmentation"].shape == (1, 3, 640, 640)


def test_detection_and_segmentation_losses_are_finite():
    model = DefectVisionModel(_config()).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 640, 640), tasks=("detection", "segmentation"))
    detection_loss = DetectionLoss(num_classes=2)(
        output["detection"],
        [{"boxes": torch.tensor([[100.0, 100.0, 300.0, 260.0]]), "labels": torch.tensor([1])}],
        levels=("P2", "P3", "P4"),
    )
    segmentation_loss = SegmentationLoss(num_classes=3)(
        output["segmentation"], torch.zeros((1, 640, 640), dtype=torch.long)
    )
    assert torch.isfinite(detection_loss["loss"])
    assert torch.isfinite(segmentation_loss["loss"])


def test_initialization_never_falls_back_when_project_checkpoint_is_missing(tmp_path):
    missing = tmp_path / "vision_base.pt"
    try:
        initialize_vision_model(_config(), mode="project_base", checkpoint=missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing project checkpoint must fail closed")
