from __future__ import annotations

import argparse
import json
from pathlib import Path



def segment_image(image_path: str | Path, checkpoint: str | Path, output_path: str | Path) -> dict:
    raise RuntimeError("semantic segmentation is blocked until a real mask manifest and validated decoder are provided")


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
