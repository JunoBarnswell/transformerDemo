from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_image import _predict_one
from .vision.model import load_vision_checkpoint
from .vision.preprocess import load_image


def detect_image(image_path: str | Path, checkpoint: str | Path) -> list[dict]:
    image = load_image(image_path)
    model, config, _ = load_vision_checkpoint(checkpoint)
    detections, _, _ = _predict_one(model, config, image)
    return [item.to_dict() for item in detections]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the detector-only visual path")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(detect_image(args.image, args.checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
