from __future__ import annotations

import argparse

from .vision.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the detection task head")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train_from_config(args.config, task_override="detection"))


if __name__ == "__main__":
    main()
