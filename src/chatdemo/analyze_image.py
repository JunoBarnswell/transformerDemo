from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .vision.preprocess import letterbox_image, restore_mask
from .vision.model import load_vision_checkpoint
from .vision.postprocess import decode_detections, render_annotated_image, save_mask_png
from .vision.tile import merge_tile_detections, merge_tile_probabilities, should_use_tiles, tile_windows
from .vision.types import Detection, ImageTransform
from .vision.preprocess import load_image


@torch.inference_mode()
def _predict_one(model, config, image: object) -> tuple[list[Detection], torch.Tensor, ImageTransform]:
    tensor, transform = letterbox_image(image, input_size=config.input_size)
    outputs = model(tensor.unsqueeze(0), tasks=("detection", "segmentation"))
    detections = decode_detections(
        outputs["detection"],
        [transform],
        reg_max=config.reg_max,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        max_detections=config.max_detections,
        class_names=config.class_names,
        levels=config.detect_levels,
    )[0]
    probabilities = outputs["segmentation"].softmax(dim=1)[0]
    restored_probabilities = restore_mask(probabilities, transform, mode="bilinear")
    return detections, restored_probabilities, transform


def _class_names_for_mask(mask: torch.Tensor, config) -> list[str]:
    class_ids = sorted(int(value) for value in mask.unique().tolist() if int(value) > 0)
    return [config.class_names[class_id - 1] if class_id - 1 < len(config.class_names) else f"mask_class_{class_id}" for class_id in class_ids]


@torch.inference_mode()
def analyze_image(
    image_path: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path = "outputs/vision",
    tile_mode: bool | str = False,
    estimated_defect_short_side: float | None = None,
) -> dict[str, Any]:
    """Run independent detection and segmentation and write original-size outputs."""
    if tile_mode not in (False, True, "auto"):
        raise ValueError("tile_mode must be false, true, or auto")
    image = load_image(image_path)
    model, config, _ = load_vision_checkpoint(checkpoint)
    image_width, image_height = image.width, image.height
    use_tiles = False
    if tile_mode is True:
        use_tiles = image_width > config.input_size or image_height > config.input_size
    elif tile_mode == "auto":
        use_tiles = should_use_tiles(
            image_width,
            image_height,
            estimated_defect_short_side=estimated_defect_short_side,
            threshold_px=config.tile_threshold_px,
            model_size=config.input_size,
        )

    if use_tiles:
        detection_parts: list[tuple[Any, Sequence[Detection]]] = []
        probability_parts: list[tuple[Any, torch.Tensor]] = []
        for window in tile_windows(image_width, image_height, tile_size=config.input_size):
            crop = image.crop((window.x0, window.y0, window.x1, window.y1))
            detections, probabilities, _ = _predict_one(model, config, crop)
            detection_parts.append((window, detections))
            probability_parts.append((window, probabilities))
        detections = merge_tile_detections(
            detection_parts,
            image_width=image_width,
            image_height=image_height,
            iou_threshold=config.nms_threshold,
            max_detections=config.max_detections,
        )
        mask = merge_tile_probabilities(
            probability_parts,
            image_width=image_width,
            image_height=image_height,
            num_classes=config.segmentation_classes,
        )
    else:
        detections, probabilities, transform = _predict_one(model, config, image)
        mask = probabilities.argmax(dim=0)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_path = save_mask_png(mask, output_path / f"{stem}_mask.png")
    annotated_path = render_annotated_image(image, mask, detections, output_path / f"{stem}_annotated.jpg")
    return {
        "image_size": [image_width, image_height],
        "coordinate_space": [config.input_size, config.input_size],
        "detections": [detection.to_dict() for detection in detections],
        "segmentation": {
            "mask_path": str(mask_path),
            "mask_size": [image_width, image_height],
            "classes": _class_names_for_mask(mask, config),
        },
        "annotated_image": str(annotated_path),
        "tile_mode": use_tiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an image with independent detection and segmentation heads")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/vision")
    parser.add_argument("--tile-mode", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--estimated-defect-short-side", type=float, default=None)
    parser.add_argument("--return-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tile_mode: bool | str = {"off": False, "on": True, "auto": "auto"}[args.tile_mode]
    result = analyze_image(
        args.image,
        args.checkpoint,
        output_dir=args.output_dir,
        tile_mode=tile_mode,
        estimated_defect_short_side=args.estimated_defect_short_side,
    )
    print(json.dumps(result, ensure_ascii=False, indent=None if args.return_json else 2))


if __name__ == "__main__":
    main()
