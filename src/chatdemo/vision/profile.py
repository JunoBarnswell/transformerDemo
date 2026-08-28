from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .model import load_vision_checkpoint


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}


@torch.inference_mode()
def estimate_flops(model: torch.nn.Module, *, input_size: int = 640) -> int:
    """Estimate multiply-add FLOPs for Conv2d/Linear layers on one input."""
    flops = [0]
    handles = []

    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d) and isinstance(output, torch.Tensor):
            batch, channels, height, width = output.shape
            kernel_h, kernel_w = module.kernel_size
            flops[0] += int(2 * batch * channels * height * width * (module.in_channels // module.groups) * kernel_h * kernel_w)
        elif isinstance(module, torch.nn.Linear) and isinstance(output, torch.Tensor):
            flops[0] += int(2 * output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(torch.zeros((1, 3, input_size, input_size), dtype=torch.float32))
    for handle in handles:
        handle.remove()
    return flops[0]


@torch.inference_mode()
def profile_model(model: torch.nn.Module, *, input_size: int = 640, iterations: int = 10, warmup: int = 2) -> dict[str, Any]:
    if input_size <= 0 or iterations <= 0 or warmup < 0:
        raise ValueError("input_size and iterations must be positive; warmup cannot be negative")
    model.eval()
    sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    for _ in range(warmup):
        model(sample)
    start = time.perf_counter()
    for _ in range(iterations):
        model(sample)
    elapsed = time.perf_counter() - start
    result = count_parameters(model)
    result["flops"] = estimate_flops(model, input_size=input_size)
    result["input_size"] = input_size
    result["iterations"] = iterations
    result["latency_ms"] = elapsed * 1000.0 / iterations
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile visual model parameters and CPU latency")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, config, _ = load_vision_checkpoint(args.checkpoint)
    result = profile_model(model, input_size=config.input_size, iterations=args.iterations, warmup=args.warmup)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
