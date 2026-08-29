from __future__ import annotations

import argparse

from .vision.train import train_from_config


def main() -> None:
    raise RuntimeError("multitask training is blocked until semantic segmentation has a validated mask dataset")


if __name__ == "__main__":
    main()
