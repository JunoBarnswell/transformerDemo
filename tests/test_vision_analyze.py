from __future__ import annotations

from pathlib import Path

from PIL import Image

from chatdemo.vision.config import VisionConfig
from chatdemo.vision.model import DefectVisionModel, save_vision_checkpoint
from chatdemo.analyze_image import analyze_image


def test_analyze_image_returns_original_size_artifacts(tmp_path: Path):
    config = VisionConfig(
        num_classes=1,
        class_names=["defect"],
        segmentation_classes=2,
        backbone_channels=(4, 8, 16, 32, 64),
        backbone_depths=(1, 1, 1, 1, 1),
        neck_channels=(4, 8, 16, 32, 64),
        use_p5=False,
    )
    checkpoint = tmp_path / "model.pt"
    save_vision_checkpoint(checkpoint, DefectVisionModel(config))
    image = tmp_path / "sample.png"
    Image.new("RGB", (1280, 720), color=(80, 90, 100)).save(image)

    result = analyze_image(image, checkpoint, output_dir=tmp_path / "outputs")
    assert result["image_size"] == [1280, 720]
    assert result["coordinate_space"] == [640, 640]
    assert Path(result["segmentation"]["mask_path"]).is_file()
    assert Path(result["annotated_image"]).is_file()
    with Image.open(result["segmentation"]["mask_path"]) as mask:
        assert mask.size == (1280, 720)
    with Image.open(result["annotated_image"]) as annotated:
        assert annotated.size == (1280, 720)
