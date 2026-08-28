from __future__ import annotations

import torch

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.detect_head import TaskAlignedAssigner
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
    assert output["detection"]["P2"]["box_logits"].shape[1] == 4 * 16
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


def test_task_aligned_positive_target_does_not_shrink_with_current_score():
    assignment = TaskAlignedAssigner(top_k=1)(
        points=torch.tensor([[5.0, 5.0]]),
        pred_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        pred_scores=torch.tensor([[0.01]]),
        gt_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        gt_labels=torch.tensor([0]),
    )
    assert assignment.foreground.tolist() == [True]
    assert assignment.target_scores[0, 0].item() == 1.0


def test_detection_head_uses_sparse_class_and_box_bias_priors():
    model = DefectVisionModel(_config())
    for level in model.config.detect_levels:
        assert torch.all(model.detection_head.cls_preds[level].bias < -4.0)
        assert torch.allclose(model.detection_head.reg_preds[level].bias, torch.full_like(model.detection_head.reg_preds[level].bias, 2.0))


def test_positive_dfl_layout_matches_decode_channel_order():
    logits = torch.arange(64 * 2 * 2, dtype=torch.float32).reshape(1, 64, 2, 2)
    raw = {"P2": {"box_logits": logits, "cls_logits": torch.zeros((1, 2, 2, 2))}}
    mask = torch.tensor([False, False, True, False])
    actual = DetectionLoss._positive_distribution(raw, 0, mask, ("P2",))
    expected = logits[0].reshape(4, 16, 2, 2).permute(2, 3, 0, 1).reshape(-1, 4, 16)[mask]
    assert torch.equal(actual, expected)
