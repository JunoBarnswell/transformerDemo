from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .backbone import Conv


STRIDES = (4, 8, 16, 32)


class DFL(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        self.conv.weight.data[:] = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, anchors = x.shape
        return self.conv(x.view(batch, 4, self.conv.in_channels, anchors).transpose(2, 1).softmax(1)).view(batch, 4, anchors)


class DetectionHead(nn.Module):
    """Native YOLOv8 v8.4.0 Detect head for P3/P4/P5 features."""

    reg_max = 16

    def __init__(self, in_channels: Sequence[int], num_classes: int, input_size: int = 640):
        super().__init__()
        if len(in_channels) not in (3, 4) or num_classes <= 0 or input_size <= 0:
            raise ValueError("DetectionHead requires three or four feature levels, positive class count and input size")
        self.nc = int(num_classes)
        self.no = self.nc + self.reg_max * 4
        c2 = max(16, int(in_channels[0]) // 4, self.reg_max * 4)
        c3 = max(int(in_channels[0]), min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for c in in_channels
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(c, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for c in in_channels
        )
        self.dfl = DFL(self.reg_max)
        self.strides = tuple(float(value) for value in STRIDES[-len(in_channels):])
        self.input_size = int(input_size)
        self.bias_init()

    def bias_init(self) -> None:
        with torch.no_grad():
            for box_head, cls_head, stride in zip(self.cv2, self.cv3, self.strides):
                box_head[-1].bias.fill_(2.0)
                cls_head[-1].bias.fill_(math.log(5.0 / self.nc / (self.input_size / stride) ** 2))

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        if len(features) != len(self.cv2):
            raise ValueError("DetectionHead feature count does not match configured architecture")
        boxes = torch.cat([head(x).flatten(2) for head, x in zip(self.cv2, features)], dim=2)
        scores = torch.cat([head(x).flatten(2) for head, x in zip(self.cv3, features)], dim=2)
        return {"boxes": boxes, "scores": scores, "feats": tuple(features)}


@dataclass(frozen=True)
class DecodedPredictions:
    boxes: torch.Tensor
    scores: torch.Tensor
    points: torch.Tensor
    strides: torch.Tensor


def make_anchors(features: Sequence[torch.Tensor], strides: Sequence[float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    strides = tuple(strides or STRIDES[-len(features):])
    points, stride_values = [], []
    dtype, device = features[0].dtype, features[0].device
    for feature, stride in zip(features, strides):
        _, _, height, width = feature.shape
        sx = torch.arange(width, device=device, dtype=dtype) + 0.5
        sy = torch.arange(height, device=device, dtype=dtype) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((sx, sy), dim=-1).reshape(-1, 2))
        stride_values.append(torch.full((height * width, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_values)


def decode_raw_predictions(raw_outputs: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]], *, reg_max: int = 16) -> DecodedPredictions:
    boxes = raw_outputs.get("boxes")
    scores = raw_outputs.get("scores")
    feats = raw_outputs.get("feats")
    if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor) or not isinstance(feats, tuple):
        raise ValueError("raw outputs must contain boxes, scores and feats from DetectionHead")
    if boxes.ndim != 3 or scores.ndim != 3 or boxes.shape[0] != scores.shape[0]:
        raise ValueError("raw detection tensors must be [B,C,A]")
    if boxes.shape[1] != 4 * reg_max:
        raise ValueError(f"box logits have {boxes.shape[1]} channels, expected {4 * reg_max}")
    points, stride_tensor = make_anchors(feats)
    distribution = boxes.permute(0, 2, 1).reshape(boxes.shape[0], -1, 4, reg_max).softmax(dim=-1)
    projection = torch.arange(reg_max, device=boxes.device, dtype=boxes.dtype)
    distances = distribution.matmul(projection) * stride_tensor.view(1, -1, 1)
    anchors = points * stride_tensor
    lt, rb = distances[..., :2], distances[..., 2:]
    decoded_boxes = torch.cat((anchors[None] - lt, anchors[None] + rb), dim=-1)
    return DecodedPredictions(decoded_boxes, scores.permute(0, 2, 1).sigmoid(), anchors, stride_tensor.squeeze(-1))


def bbox_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes1.shape != boxes2.shape or boxes1.shape[-1] != 4:
        raise ValueError("boxes must have equal [...,4] shapes")
    w1 = boxes1[..., 2] - boxes1[..., 0]
    h1 = boxes1[..., 3] - boxes1[..., 1] + eps
    w2 = boxes2[..., 2] - boxes2[..., 0]
    h2 = boxes2[..., 3] - boxes2[..., 1] + eps
    inter = (torch.minimum(boxes1[..., 2], boxes2[..., 2]) - torch.maximum(boxes1[..., 0], boxes2[..., 0])).clamp_min(0) * (torch.minimum(boxes1[..., 3], boxes2[..., 3]) - torch.maximum(boxes1[..., 1], boxes2[..., 1])).clamp_min(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.maximum(boxes1[..., 2], boxes2[..., 2]) - torch.minimum(boxes1[..., 0], boxes2[..., 0])
    ch = torch.maximum(boxes1[..., 3], boxes2[..., 3]) - torch.minimum(boxes1[..., 1], boxes2[..., 1])
    c2 = cw.square() + ch.square() + eps
    rho2 = ((boxes2[..., 0] + boxes2[..., 2] - boxes1[..., 0] - boxes1[..., 2]).square() + (boxes2[..., 1] + boxes2[..., 3] - boxes1[..., 1] - boxes1[..., 3]).square()) / 4
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).square()
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


@dataclass(frozen=True)
class Assignment:
    labels: torch.Tensor
    boxes: torch.Tensor
    target_scores: torch.Tensor
    foreground: torch.Tensor
    matched_gt: torch.Tensor


class TaskAlignedAssigner:
    """YOLOv8 v8.4.0 TAL with explicit batched tensors and no hidden fallback."""

    def __init__(self, top_k: int = 10, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        self.top_k, self.alpha, self.beta, self.eps = int(top_k), float(alpha), float(beta), float(eps)

    @torch.no_grad()
    def __call__(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor, points: torch.Tensor, gt_labels: torch.Tensor, gt_boxes: torch.Tensor, mask_gt: torch.Tensor | None = None) -> Assignment:
        if pred_scores.ndim != 3 or pred_boxes.ndim != 3 or gt_labels.ndim != 3 or gt_boxes.ndim != 3:
            raise ValueError("TAL expects [B,A,C], [B,A,4], [B,M,1], [B,M,4]")
        batch, anchors, classes = pred_scores.shape
        max_gt = gt_boxes.shape[1]
        if mask_gt is None:
            mask_gt = torch.ones((batch, max_gt, 1), dtype=torch.bool, device=gt_boxes.device)
        if max_gt == 0:
            return Assignment(torch.full((batch, anchors), -1, dtype=torch.long, device=gt_boxes.device), torch.zeros_like(pred_boxes), torch.zeros_like(pred_scores), torch.zeros((batch, anchors), dtype=torch.bool, device=gt_boxes.device), torch.full((batch, anchors), -1, dtype=torch.long, device=gt_boxes.device))
        in_gts = self.select_candidates_in_gts(points, gt_boxes, mask_gt)
        overlaps = torch.zeros((batch, max_gt, anchors), dtype=pred_boxes.dtype, device=pred_boxes.device)
        scores = torch.zeros_like(overlaps)
        for b in range(batch):
            for g in range(max_gt):
                if not mask_gt[b, g, 0]:
                    continue
                scores[b, g] = pred_scores[b, :, int(gt_labels[b, g, 0])]
                overlaps[b, g] = bbox_ciou(pred_boxes[b], gt_boxes[b, g].expand_as(pred_boxes[b])).clamp_min(0)
        alignment = scores.clamp_min(0).pow(self.alpha) * overlaps.pow(self.beta)
        alignment = alignment.masked_fill(~in_gts, 0)
        k = min(self.top_k, anchors)
        _, top_indices = alignment.topk(k, dim=-1)
        mask_pos = torch.zeros_like(alignment, dtype=torch.int8)
        mask_pos.scatter_add_(-1, top_indices, torch.ones_like(top_indices, dtype=torch.int8))
        mask_pos.masked_fill_(mask_pos > 1, 0)
        valid_gt = mask_gt.squeeze(-1)[:, :, None].expand(-1, max_gt, anchors)
        mask_pos = mask_pos.to(alignment.dtype) * in_gts.to(alignment.dtype) * valid_gt.to(alignment.dtype)
        fg_count = mask_pos.sum(dim=1)
        if fg_count.max() > 1:
            best_gt = overlaps.argmax(dim=1)
            keep = torch.zeros_like(mask_pos)
            keep.scatter_(1, best_gt[:, None], 1)
            mask_pos = torch.where(fg_count[:, None] > 1, keep, mask_pos)
        foreground = mask_pos.sum(dim=1).bool()
        matched_gt = mask_pos.argmax(dim=1)
        labels = gt_labels.squeeze(-1).gather(1, matched_gt)
        target_boxes = gt_boxes.gather(1, matched_gt[..., None].expand(-1, -1, 4))
        one_hot = torch.zeros_like(pred_scores)
        one_hot.scatter_(2, labels[..., None], 1)
        one_hot *= foreground[..., None].to(one_hot.dtype)
        normalized = (alignment * mask_pos)
        max_alignment = normalized.amax(dim=-1, keepdim=True)
        max_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        quality = (normalized * max_overlap / (max_alignment + self.eps)).amax(dim=1)
        return Assignment(labels, target_boxes, one_hot * quality[..., None], foreground, matched_gt)

    @staticmethod
    def select_candidates_in_gts(points: torch.Tensor, gt_boxes: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        boxes = gt_boxes.clone()
        wh = boxes[..., 2:] - boxes[..., :2]
        small = wh < 8.0
        center = (boxes[..., :2] + boxes[..., 2:]) / 2
        adjusted_wh = torch.where(small & mask_gt.bool(), torch.full_like(wh, 16.0), wh)
        boxes = torch.cat((center - adjusted_wh / 2, center + adjusted_wh / 2), dim=-1)
        deltas = torch.cat((points[None, None] - boxes[..., None, :2], boxes[..., None, 2:] - points[None, None]), dim=-1)
        return deltas.amin(dim=-1).gt(1e-9)
