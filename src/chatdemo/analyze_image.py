from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .vision.preprocess import letterbox_image
from .vision.model import load_vision_checkpoint
from .vision.postprocess import decode_detections
from .vision.types import Detection, ImageTransform
from .vision.preprocess import load_image


@torch.inference_mode()
def _predict_one(model, config, image: object) -> tuple[list[Detection], ImageTransform]:
    tensor, transform = letterbox_image(image, input_size=config.input_size)
    outputs = model(tensor.unsqueeze(0), tasks=("detection",))
    detections = decode_detections(
        outputs["detection"],
        [transform],
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        max_detections=config.max_detections,
        class_names=config.class_names,
    )[0]
    return detections, transform


@torch.inference_mode()
def analyze_image(
    image_path: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path = "outputs/vision",
    tile_mode: bool | str = False,
    estimated_defect_short_side: float | None = None,
) -> dict[str, Any]:
    """Run detector and return canonical 640-coordinate boxes."""
    if tile_mode not in (False, True, "auto"):
        raise ValueError("tile_mode is unavailable until the validated P2 tiler is added")
    image = load_image(image_path)
    model, config, _ = load_vision_checkpoint(checkpoint)
    image_width, image_height = image.width, image.height
    if tile_mode is not False:
        raise ValueError("tile_mode must be false; tiling is blocked until P2 validation")
    detections, _ = _predict_one(model, config, image)
    return {
        "image_size": [image_width, image_height],
        "coordinate_space": [config.input_size, config.input_size],
        "detections": [detection.to_dict() for detection in detections],
        "tile_mode": False,
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
