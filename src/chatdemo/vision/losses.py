from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coordinates import box_area
from .detect_head import TaskAlignedAssigner, decode_raw_predictions


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Complete-IoU loss for xyxy boxes."""
    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("pred_boxes and target_boxes must have the same [...,4] shape")
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    pred_wh = (pred_boxes[..., 2:] - pred_boxes[..., :2]).clamp_min(1e-6)
    target_wh = (target_boxes[..., 2:] - target_boxes[..., :2]).clamp_min(1e-6)
    inter_lt = torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
    inter_rb = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp_min(0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]
    union = box_area(pred_boxes) + box_area(target_boxes) - intersection
    iou = intersection / union.clamp_min(1e-7)

    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
    center_distance = ((pred_center - target_center) ** 2).sum(dim=-1)
    enclose_lt = torch.minimum(pred_boxes[..., :2], target_boxes[..., :2])
    enclose_rb = torch.maximum(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp_min(1e-6)
    diagonal = (enclose_wh ** 2).sum(dim=-1).clamp_min(1e-7)

    pred_angle = torch.atan(pred_wh[..., 0] / pred_wh[..., 1])
    target_angle = torch.atan(target_wh[..., 0] / target_wh[..., 1])
    v = (4.0 / math.pi**2) * (target_angle - pred_angle).pow(2)
    alpha = v / (1.0 - iou + v).clamp_min(1e-7)
    return 1.0 - iou + center_distance / diagonal + alpha * v


def distribution_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Return per-anchor DFL for four left/top/right/bottom distances."""
    if pred_logits.ndim != 3 or pred_logits.shape[-1] != reg_max:
        raise ValueError("pred_logits must have shape [N,4,reg_max]")
    if target.shape != pred_logits.shape[:2]:
        raise ValueError("DFL target must have shape [N,4]")
    if pred_logits.numel() == 0:
        return pred_logits.sum() * 0.0
    target = target.clamp(0.0, float(reg_max - 1) - 1e-2)
    left = target.floor().long()
    right = (left + 1).clamp_max(reg_max - 1)
    right_weight = target - left.float()
    left_weight = 1.0 - right_weight
    flat_logits = pred_logits.reshape(-1, reg_max)
    left_loss = F.cross_entropy(flat_logits, left.reshape(-1), reduction="none").reshape_as(target)
    right_loss = F.cross_entropy(flat_logits, right.reshape(-1), reduction="none").reshape_as(target)
    return (left_loss * left_weight + right_loss * right_weight).mean(dim=-1)


class DetectionLoss(nn.Module):
    """Task-aligned classification + CIoU + DFL loss."""

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        cls_weight: float = 0.5,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        assigner: TaskAlignedAssigner | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.dfl_weight = float(dfl_weight)
        self.assigner = assigner or TaskAlignedAssigner()

    def forward(
        self,
        raw_outputs: Mapping[str, Mapping[str, torch.Tensor]],
        targets: Sequence[Mapping[str, torch.Tensor]],
        levels: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        decoded = decode_raw_predictions(raw_outputs, reg_max=self.reg_max, levels=levels)
        batch_size = decoded.boxes.shape[0]
        if len(targets) != batch_size:
            raise ValueError("one detection target is required per batch item")

        zero = decoded.boxes.sum() * 0.0
        cls_numerator = zero
        box_numerator = zero
        dfl_numerator = zero
        quality_sum = zero
        positive_count = 0
        for batch_index in range(batch_size):
            gt_boxes = targets[batch_index].get("boxes")
            gt_labels = targets[batch_index].get("labels")
            if gt_boxes is None or gt_labels is None:
                gt_boxes = decoded.boxes.new_zeros((0, 4))
                gt_labels = torch.zeros((0,), dtype=torch.long, device=decoded.boxes.device)
            gt_boxes = gt_boxes.to(device=decoded.boxes.device, dtype=decoded.boxes.dtype).reshape(-1, 4)
            gt_labels = gt_labels.to(device=decoded.boxes.device, dtype=torch.long).reshape(-1)
            assignment = self.assigner(
                decoded.points,
                decoded.boxes[batch_index].detach(),
                decoded.scores[batch_index].detach(),
                gt_boxes,
                gt_labels,
            )
            target_scores = assignment.target_scores
            cls_logits = self._classification_logits(raw_outputs, batch_index, levels)
            cls_numerator = cls_numerator + F.binary_cross_entropy_with_logits(
                cls_logits,
                target_scores,
                reduction="sum",
            )
            quality_sum = quality_sum + target_scores.sum()
            if assignment.foreground.any():
                positive = assignment.foreground
                positive_count += int(positive.sum().item())
                quality_weight = target_scores[positive].sum(dim=-1)
                box_numerator = box_numerator + (
                    ciou_loss(decoded.boxes[batch_index, positive], assignment.boxes[positive]) * quality_weight
                ).sum()
                distances = self._target_distances(decoded.points[positive], assignment.boxes[positive], decoded.strides[positive])
                pred_distribution = self._positive_distribution(raw_outputs, batch_index, positive, levels)
                dfl_numerator = dfl_numerator + (
                    distribution_focal_loss(pred_distribution, distances, self.reg_max) * quality_weight
                ).sum()

        denominator = quality_sum.clamp_min(1.0)
        cls_loss = cls_numerator / denominator
        box_loss = box_numerator / denominator
        dfl_loss = dfl_numerator / denominator
        total = self.cls_weight * cls_loss + self.box_weight * box_loss + self.dfl_weight * dfl_loss
        return {
            "loss": total,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
            "quality_sum": quality_sum.detach(),
            "positive_count": torch.tensor(float(positive_count), device=total.device),
        }

    @staticmethod
    def _target_distances(points: torch.Tensor, boxes: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
        distances = torch.stack(
            (
                points[:, 0] - boxes[:, 0],
                points[:, 1] - boxes[:, 1],
                boxes[:, 2] - points[:, 0],
                boxes[:, 3] - points[:, 1],
            ),
            dim=-1,
        )
        return distances / strides[:, None].clamp_min(1e-6)

    @staticmethod
    def _positive_distribution(
        raw_outputs: Mapping[str, Mapping[str, torch.Tensor]],
        batch_index: int,
        positive_mask: torch.Tensor,
        levels: Sequence[str] | None,
    ) -> torch.Tensor:
        selected_levels = tuple(levels or raw_outputs.keys())
        distribution_by_level = []
        for level in selected_levels:
            logits = raw_outputs[level]["box_logits"][batch_index]
            channels, height, width = logits.shape
            bins = channels // 4
            values = logits.reshape(4, bins, height, width).permute(2, 3, 0, 1).reshape(-1, 4, bins)
            distribution_by_level.append(values)
        distribution = torch.cat(distribution_by_level, dim=0)
        return distribution[positive_mask]

    @staticmethod
    def _classification_logits(
        raw_outputs: Mapping[str, Mapping[str, torch.Tensor]],
        batch_index: int,
        levels: Sequence[str] | None,
    ) -> torch.Tensor:
        selected_levels = tuple(levels or raw_outputs.keys())
        values = []
        for level in selected_levels:
            logits = raw_outputs[level]["cls_logits"][batch_index]
            values.append(logits.permute(1, 2, 0).reshape(-1, logits.shape[0]))
        return torch.cat(values, dim=0)


class SegmentationLoss(nn.Module):
    """Cross-entropy plus soft Dice for semantic masks."""

    def __init__(self, num_classes: int, dice_weight: float = 1.0, ignore_index: int = 255, focal_gamma: float | None = None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("segmentation requires at least two classes")
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("segmentation logits must be BCHW and target must be BHW")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("segmentation logits and target have incompatible shapes")
        target = target.to(device=logits.device, dtype=torch.long)
        valid = target != self.ignore_index
        if valid.any() and ((target[valid] < 0) | (target[valid] >= self.num_classes)).any():
            raise ValueError("segmentation target contains a class outside the configured range")
        if not valid.any():
            zero = logits.sum() * 0.0
            return {"loss": zero, "ce_loss": zero, "dice_loss": zero}
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        probabilities = logits.softmax(dim=1)
        safe_target = target.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid[:, None].float()
        intersection = (probabilities * one_hot * valid_mask).sum(dim=(0, 2, 3))
        denominator = (probabilities * valid_mask).sum(dim=(0, 2, 3)) + (one_hot * valid_mask).sum(dim=(0, 2, 3))
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        if self.focal_gamma is not None:
            log_prob = F.log_softmax(logits, dim=1)
            gathered = log_prob.gather(1, safe_target[:, None]).squeeze(1)
            focal = (((1.0 - gathered.exp()).clamp_min(0.0) ** float(self.focal_gamma)) * (-gathered) * valid).sum() / valid.sum().clamp_min(1)
            ce = focal
        total = ce + self.dice_weight * dice
        return {"loss": total, "ce_loss": ce, "dice_loss": dice}
