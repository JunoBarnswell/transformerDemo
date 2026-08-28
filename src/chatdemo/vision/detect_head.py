from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn

from .backbone import ConvBNAct
from .coordinates import box_iou


STRIDES: Dict[str, int] = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}


def _channels_for_levels(in_channels: int | Mapping[str, int] | Sequence[int], levels: Sequence[str]) -> Dict[str, int]:
    if isinstance(in_channels, int):
        return {level: in_channels for level in levels}
    if isinstance(in_channels, Mapping):
        missing = [level for level in levels if level not in in_channels]
        if missing:
            raise ValueError(f"detection head channels missing levels: {missing}")
        return {level: int(in_channels[level]) for level in levels}
    values = list(in_channels)
    if len(values) != len(levels):
        raise ValueError("sequence in_channels must have one value per detection level")
    return {level: int(value) for level, value in zip(levels, values)}


class DetectionHead(nn.Module):
    """Anchor-free decoupled detection head with DFL distributions."""

    def __init__(
        self,
        in_channels: int | Mapping[str, int] | Sequence[int],
        num_classes: int,
        levels: Sequence[str] = ("P2", "P3", "P4"),
        reg_max: int = 16,
        hidden_channels: int = 64,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if reg_max <= 0:
            raise ValueError("reg_max must be positive")
        if not levels or any(level not in STRIDES for level in levels):
            raise ValueError("levels must be drawn from P2/P3/P4/P5")
        self.num_classes = int(num_classes)
        self.levels = tuple(levels)
        self.reg_max = int(reg_max)
        self.channel_by_level = _channels_for_levels(in_channels, self.levels)

        self.stems = nn.ModuleDict()
        self.cls_towers = nn.ModuleDict()
        self.reg_towers = nn.ModuleDict()
        self.cls_preds = nn.ModuleDict()
        self.reg_preds = nn.ModuleDict()
        for level in self.levels:
            self.stems[level] = ConvBNAct(self.channel_by_level[level], hidden_channels, 3)
            self.cls_towers[level] = nn.Sequential(ConvBNAct(hidden_channels, hidden_channels, 3), ConvBNAct(hidden_channels, hidden_channels, 3))
            self.reg_towers[level] = nn.Sequential(ConvBNAct(hidden_channels, hidden_channels, 3), ConvBNAct(hidden_channels, hidden_channels, 3))
            self.cls_preds[level] = nn.Conv2d(hidden_channels, self.num_classes, 1)
            self.reg_preds[level] = nn.Conv2d(hidden_channels, 4 * (self.reg_max + 1), 1)

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        missing = [level for level in self.levels if level not in features]
        if missing:
            raise ValueError(f"features missing detection levels: {missing}")
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for level in self.levels:
            shared = self.stems[level](features[level])
            outputs[level] = {
                "box_logits": self.reg_preds[level](self.reg_towers[level](shared)),
                "cls_logits": self.cls_preds[level](self.cls_towers[level](shared)),
            }
        return outputs


@dataclass(frozen=True)
class DecodedPredictions:
    boxes: torch.Tensor
    scores: torch.Tensor
    points: torch.Tensor
    strides: torch.Tensor


def _grid(height: int, width: int, stride: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).reshape(-1, 2)


def decode_raw_predictions(
    raw_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    reg_max: int,
    levels: Sequence[str] | None = None,
) -> DecodedPredictions:
    """Decode DFL logits into model-input xyxy boxes and class scores."""
    selected_levels = tuple(levels or raw_outputs.keys())
    boxes_by_level = []
    scores_by_level = []
    points_by_level = []
    strides_by_level = []
    bins = None
    for level in selected_levels:
        if level not in raw_outputs:
            raise ValueError(f"raw outputs missing level: {level}")
        box_logits = raw_outputs[level]["box_logits"]
        cls_logits = raw_outputs[level]["cls_logits"]
        if box_logits.ndim != 4 or cls_logits.ndim != 4:
            raise ValueError("raw detection outputs must be BCHW tensors")
        batch, channels, height, width = box_logits.shape
        expected_channels = 4 * (reg_max + 1)
        if channels != expected_channels:
            raise ValueError(f"{level} box logits have {channels} channels, expected {expected_channels}")
        if cls_logits.shape[0] != batch or cls_logits.shape[-2:] != (height, width):
            raise ValueError("box and class logits have incompatible shapes")
        stride = STRIDES[level]
        if bins is None:
            bins = torch.arange(reg_max + 1, device=box_logits.device, dtype=box_logits.dtype)
        distribution = box_logits.reshape(batch, 4, reg_max + 1, height, width).softmax(dim=2)
        distances = (distribution * bins.view(1, 1, -1, 1, 1)).sum(dim=2) * stride
        distances = distances.permute(0, 2, 3, 1).reshape(batch, -1, 4)
        points = _grid(height, width, stride, box_logits.device, box_logits.dtype)
        center_x = points[:, 0].view(1, -1)
        center_y = points[:, 1].view(1, -1)
        boxes = torch.stack(
            (
                center_x - distances[..., 0],
                center_y - distances[..., 1],
                center_x + distances[..., 2],
                center_y + distances[..., 3],
            ),
            dim=-1,
        )
        scores = cls_logits.sigmoid().permute(0, 2, 3, 1).reshape(batch, -1, cls_logits.shape[1])
        boxes_by_level.append(boxes)
        scores_by_level.append(scores)
        points_by_level.append(points)
        strides_by_level.append(torch.full((points.shape[0],), float(stride), device=points.device, dtype=points.dtype))

    if not boxes_by_level:
        raise ValueError("at least one detection level is required")
    return DecodedPredictions(
        boxes=torch.cat(boxes_by_level, dim=1),
        scores=torch.cat(scores_by_level, dim=1),
        points=torch.cat(points_by_level, dim=0),
        strides=torch.cat(strides_by_level, dim=0),
    )


@dataclass(frozen=True)
class Assignment:
    foreground: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    target_scores: torch.Tensor
    matched_gt: torch.Tensor


class TaskAlignedAssigner:
    """Small, explicit Task-Aligned Assigner for anchor-free predictions."""

    def __init__(self, top_k: int = 10, alpha: float = 0.5, beta: float = 6.0):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def __call__(
        self,
        points: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Assignment:
        if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes must have shape [N,4]")
        if pred_scores.ndim != 2 or pred_scores.shape[0] != pred_boxes.shape[0]:
            raise ValueError("pred_scores must have shape [N,C]")
        if gt_boxes.numel() == 0:
            return Assignment(
                foreground=torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device),
                labels=torch.full((pred_boxes.shape[0],), -1, dtype=torch.long, device=pred_boxes.device),
                boxes=torch.zeros_like(pred_boxes),
                target_scores=torch.zeros_like(pred_scores),
                matched_gt=torch.full((pred_boxes.shape[0],), -1, dtype=torch.long, device=pred_boxes.device),
            )
        if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4 or gt_labels.ndim != 1 or gt_labels.shape[0] != gt_boxes.shape[0]:
            raise ValueError("gt boxes and labels have incompatible shapes")
        if (gt_labels < 0).any() or (gt_labels >= pred_scores.shape[1]).any():
            raise ValueError("gt label is outside the detector class range")

        ious = box_iou(pred_boxes, gt_boxes)
        class_scores = pred_scores[:, gt_labels]
        alignment = class_scores.clamp_min(1e-8).pow(self.alpha) * ious.clamp_min(0).pow(self.beta)
        inside = (
            (points[:, None, 0] >= gt_boxes[None, :, 0])
            & (points[:, None, 0] <= gt_boxes[None, :, 2])
            & (points[:, None, 1] >= gt_boxes[None, :, 1])
            & (points[:, None, 1] <= gt_boxes[None, :, 3])
        )
        alignment = alignment.masked_fill(~inside, -1.0)

        foreground = torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device)
        matched_gt = torch.full((pred_boxes.shape[0],), -1, dtype=torch.long, device=pred_boxes.device)
        matched_metric = torch.full((pred_boxes.shape[0],), -1.0, device=pred_boxes.device)
        for gt_index in range(gt_boxes.shape[0]):
            candidate_metric = alignment[:, gt_index]
            valid = candidate_metric >= 0
            if not valid.any():
                continue
            candidate_indices = torch.where(valid)[0]
            k = min(self.top_k, candidate_indices.numel())
            values, local_indices = torch.topk(candidate_metric[candidate_indices], k=k)
            for value, local_index in zip(values, local_indices):
                anchor_index = candidate_indices[local_index]
                if value > matched_metric[anchor_index]:
                    matched_metric[anchor_index] = value
                    matched_gt[anchor_index] = gt_index
                    foreground[anchor_index] = True

        labels = torch.full((pred_boxes.shape[0],), -1, dtype=torch.long, device=pred_boxes.device)
        boxes = torch.zeros_like(pred_boxes)
        target_scores = torch.zeros_like(pred_scores)
        if foreground.any():
            indices = torch.where(foreground)[0]
            gt_indices = matched_gt[indices]
            labels[indices] = gt_labels[gt_indices]
            boxes[indices] = gt_boxes[gt_indices]
            target_scores[indices, labels[indices]] = (
                ious[indices, gt_indices].clamp(0.0, 1.0) * pred_scores[indices, labels[indices]].detach().clamp(0.0, 1.0)
            ).clamp_min(1e-3)
        return Assignment(foreground, labels, boxes, target_scores, matched_gt)
