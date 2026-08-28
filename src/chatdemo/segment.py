from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_image import _predict_one
from .vision.model import load_vision_checkpoint
from .vision.postprocess import save_mask_png
from .vision.preprocess import load_image


def segment_image(image_path: str | Path, checkpoint: str | Path, output_path: str | Path) -> dict:
    image = load_image(image_path)
    model, config, _ = load_vision_checkpoint(checkpoint)
    _, probabilities, _ = _predict_one(model, config, image)
    mask_path = save_mask_png(probabilities.argmax(dim=0), output_path)
    return {
        "mask_path": str(mask_path),
        "mask_size": [image.width, image.height],
        "classes": sorted(int(value) for value in probabilities.argmax(dim=0).unique().tolist()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the segmentation-only visual path")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/vision/mask.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(segment_image(args.image, args.checkpoint, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
