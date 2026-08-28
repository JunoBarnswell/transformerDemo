from __future__ import annotations

import pytest

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.fewshot import apply_fewshot_stage
from chatdemo.vision.model import DefectVisionModel


def _model() -> DefectVisionModel:
    return DefectVisionModel(VisionConfig(
        num_classes=1,
        class_names=["defect"],
        backbone_channels=(4, 8, 16, 32, 64),
        backbone_depths=(1, 1, 1, 1, 1),
        neck_channels=(4, 8, 16, 32, 64),
        use_p5=False,
    ))


def test_f1_freezes_backbone_and_regression_but_keeps_classification_trainable():
    model = apply_fewshot_stage(_model(), "f1")
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.detection_head.reg_preds.parameters())
    assert any(parameter.requires_grad for parameter in model.detection_head.cls_preds.parameters())


def test_f0_is_a_checkpoint_stage_not_an_accidental_training_fallback():
    with pytest.raises(ValueError, match="no trainable"):
        apply_fewshot_stage(_model(), "f0")
