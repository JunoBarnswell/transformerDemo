from __future__ import annotations

import argparse

from .vision.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning with explicit D1-D3 stage selection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("d1", "d2", "d3"), required=True)
    args = parser.parse_args()
    print(train_from_config(args.config, stage_override=args.stage))


if __name__ == "__main__":
    main()
