from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class Conv(nn.Module):
    """YOLOv8 Conv: Conv2d (no bias) + BN + SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 3, 1)
        self.cv2 = Conv(hidden, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        parts.extend(block(parts[-1]) for block in self.m)
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1, act=False)
        self.cv2 = Conv(hidden * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [self.cv1(x)]
        pooled.extend(self.m(pooled[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(pooled, dim=1))
        return y + x if self.add else y


class Concat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(tensors), self.d)
