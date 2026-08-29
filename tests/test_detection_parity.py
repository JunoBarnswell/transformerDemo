from __future__ import annotations

import pytest
import torch

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.detect_head import TaskAlignedAssigner, bbox_ciou, decode_raw_predictions
from chatdemo.vision.losses import distribution_focal_loss
from chatdemo.vision.model import DefectVisionModel


def _model(classes: int = 80) -> DefectVisionModel:
    return DefectVisionModel(VisionConfig(num_classes=classes, class_names=[f"c{i}" for i in range(classes)]))


def test_graph_state_and_forward_match_ultralytics_v840():
    ultralytics = pytest.importorskip("ultralytics")
    if getattr(ultralytics, "__version__", None) != "8.4.0":
        pytest.skip("golden parity requires ultralytics 8.4.0")
    from ultralytics import YOLO

    official = YOLO("yolov8n.yaml").model.eval()
    native = _model().eval()
    native.load_state_dict(official.state_dict(), strict=True)
    torch.manual_seed(7)
    image = torch.randn(1, 3, 640, 640)
    official_output = official(image)[0].clone()
    official_output[:, :4] = torch.cat((official_output[:, :2] - official_output[:, 2:4] / 2, official_output[:, :2] + official_output[:, 2:4] / 2), dim=1)
    native_output = native(image)["detection"]
    native_decoded = decode_raw_predictions(native_output)
    native_output_tensor = torch.cat((native_decoded.boxes.permute(0, 2, 1), native_decoded.scores.permute(0, 2, 1)), dim=1)
    assert torch.allclose(native_output_tensor, official_output, atol=1e-4, rtol=1e-4)


def test_dfl_and_ciou_match_official_reference():
    ultralytics = pytest.importorskip("ultralytics")
    if getattr(ultralytics, "__version__", None) != "8.4.0":
        pytest.skip("golden parity requires ultralytics 8.4.0")
    from ultralytics.utils.loss import DFLoss
    from ultralytics.utils.metrics import bbox_iou

    torch.manual_seed(3)
    logits = torch.randn(5, 4, 16)
    target = torch.rand(5, 4) * 15
    expected_dfl = DFLoss(16)(logits.reshape(-1, 16), target).reshape(-1)
    assert torch.allclose(distribution_focal_loss(logits, target), expected_dfl, atol=1e-6, rtol=1e-6)
    first = torch.rand(5, 4) * 100
    second = torch.rand(5, 4) * 100
    boxes1 = torch.cat((torch.minimum(first[:, :2], first[:, 2:]), torch.maximum(first[:, :2], first[:, 2:])), dim=1)
    boxes2 = torch.cat((torch.minimum(second[:, :2], second[:, 2:]), torch.maximum(second[:, :2], second[:, 2:])), dim=1)
    expected_ciou = bbox_iou(boxes1, boxes2, xywh=False, CIoU=True).squeeze(-1)
    assert torch.allclose(bbox_ciou(boxes1, boxes2), expected_ciou, atol=1e-6, rtol=1e-6)


def test_tal_matches_official_reference():
    ultralytics = pytest.importorskip("ultralytics")
    if getattr(ultralytics, "__version__", None) != "8.4.0":
        pytest.skip("golden parity requires ultralytics 8.4.0")
    from ultralytics.utils.tal import TaskAlignedAssigner as OfficialTAL

    torch.manual_seed(11)
    scores = torch.rand(1, 20, 3)
    boxes = torch.rand(1, 20, 4) * 640
    boxes = torch.cat((torch.minimum(boxes[..., :2], boxes[..., 2:]), torch.maximum(boxes[..., :2], boxes[..., 2:])), dim=-1)
    points = torch.rand(20, 2) * 640
    labels = torch.tensor([[[0], [2]]])
    gt = torch.tensor([[[100.0, 100.0, 340.0, 340.0], [300.0, 300.0, 500.0, 500.0]]])
    mask = torch.ones((1, 2, 1), dtype=torch.bool)
    ours = TaskAlignedAssigner()(scores, boxes, points, labels, gt, mask)
    official = OfficialTAL(topk=10, num_classes=3, alpha=0.5, beta=6.0, stride=[8, 16, 32])(
        scores, boxes, points, labels, gt, mask
    )
    assert torch.equal(ours.labels, official[0])
    assert torch.allclose(ours.boxes, official[1], atol=1e-6)
    assert torch.allclose(ours.target_scores, official[2], atol=1e-6)
    assert torch.equal(ours.foreground, official[3].bool())
    assert torch.equal(ours.matched_gt, official[4])


def test_p2_graph_matches_official_state_and_has_34000_anchors():
    ultralytics = pytest.importorskip("ultralytics")
    if getattr(ultralytics, "__version__", None) != "8.4.0":
        pytest.skip("golden parity requires ultralytics 8.4.0")
    from ultralytics import YOLO

    official = YOLO("yolov8n-p2.yaml").model.eval()
    native = DefectVisionModel(VisionConfig(architecture="yolov8n-p2", num_classes=80, class_names=[f"c{i}" for i in range(80)])).eval()
    native.load_state_dict(official.state_dict(), strict=True)
    output = native(torch.zeros(1, 3, 640, 640))["detection"]
    assert output["boxes"].shape == (1, 64, 34000)


def test_checkpoint_format_is_architecture_strict(tmp_path):
    from chatdemo.vision.model import load_vision_checkpoint, save_vision_checkpoint

    model = _model(2)
    path = save_vision_checkpoint(tmp_path / "best.pt", model)
    loaded, config, payload = load_vision_checkpoint(path)
    assert config.architecture == "yolov8n"
    assert payload["format"] == "chatdemo-detection-v3"
    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_segmentation_path_fails_closed():
    from chatdemo.segment import segment_image

    with pytest.raises(RuntimeError, match="blocked"):
        segment_image("missing.jpg", "missing.pt", "mask.png")


def test_p2_conversion_copies_shared_geometry_and_reinitializes_new_branch(tmp_path):
    from tools.extend_p2_checkpoint import convert
    from chatdemo.vision.model import load_vision_checkpoint, save_vision_checkpoint

    source = save_vision_checkpoint(tmp_path / "source.pt", _model(2))
    target = convert(source, tmp_path / "p2.pt")
    loaded, config, payload = load_vision_checkpoint(target)
    assert config.architecture == "yolov8n-p2"
    assert payload["extra"]["conversion"] == "explicit-yolov8n-to-yolov8n-p2"
    assert loaded(torch.zeros(1, 3, 640, 640))["detection"]["boxes"].shape[-1] == 34000
