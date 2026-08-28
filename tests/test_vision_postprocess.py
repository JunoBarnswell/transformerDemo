from __future__ import annotations

import torch

from chatdemo.vision.postprocess import decode_detections, nms
from chatdemo.vision.preprocess import build_image_transform


def test_nms_removes_overlapping_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]])
    keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7]), iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_decoded_detector_boxes_are_canonical_not_letterbox_coordinates():
    # Force a deterministic distribution at a point inside the non-padded
    # region of a wide letterboxed image.
    box_logits = torch.full((1, 64, 160, 160), -10.0)
    for coordinate in range(4):
        box_logits[:, coordinate * 16 + 2, 40, 40] = 10.0
    cls_logits = torch.full((1, 2, 160, 160), -10.0)
    cls_logits[:, 0, 40, 40] = 10.0
    transform = build_image_transform(1280, 720)
    result = decode_detections(
        {"P2": {"box_logits": box_logits, "cls_logits": cls_logits}},
        [transform],
        reg_max=16,
        confidence_threshold=0.5,
        levels=("P2",),
        class_names=["scratch", "pit"],
    )[0]
    assert len(result) == 1
    assert result[0].class_name == "scratch"
    assert all(0.0 <= value <= 640.0 for value in result[0].box_xyxy)
