from __future__ import annotations

import argparse

from .vision.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning with explicit F0-F4 stage selection")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("f0", "f1", "f2", "f3", "f4"), required=True)
    parser.add_argument("--task", choices=("detection", "segmentation", "multitask"), default=None)
    args = parser.parse_args()
    print(train_from_config(args.config, task_override=args.task, fewshot_stage_override=args.stage))


if __name__ == "__main__":
    main()
