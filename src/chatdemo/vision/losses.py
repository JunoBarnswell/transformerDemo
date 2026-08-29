from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detect_head import TaskAlignedAssigner, bbox_ciou, decode_raw_predictions


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    return 1.0 - bbox_ciou(pred_boxes, target_boxes)


def distribution_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    if pred_logits.ndim != 3 or pred_logits.shape[-1] != reg_max or target.shape != pred_logits.shape[:2]:
        raise ValueError("DFL expects logits [N,4,reg_max] and target [N,4]")
    if pred_logits.numel() == 0:
        return pred_logits.sum() * 0.0
    target = target.clamp(0, reg_max - 1 - 0.01)
    left = target.long()
    right = left + 1
    wl = right - target
    wr = 1 - wl
    flat = pred_logits.reshape(-1, reg_max)
    return (F.cross_entropy(flat, left.reshape(-1), reduction="none").reshape_as(target) * wl + F.cross_entropy(flat, right.reshape(-1), reduction="none").reshape_as(target) * wr).mean(dim=-1)


class DetectionLoss(nn.Module):
    """YOLOv8 v8.4.0 detection loss with fixed mathematical contract."""

    def __init__(self, num_classes: int, cls_weight: float = 0.5, box_weight: float = 7.5, dfl_weight: float = 1.5, assigner: TaskAlignedAssigner | None = None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.cls_weight, self.box_weight, self.dfl_weight = float(cls_weight), float(box_weight), float(dfl_weight)
        self.assigner = assigner or TaskAlignedAssigner(top_k=10, alpha=0.5, beta=6.0)

    def forward(self, raw_outputs: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]], targets: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        decoded = decode_raw_predictions(raw_outputs)
        batch_size, anchors = decoded.boxes.shape[:2]
        if len(targets) != batch_size:
            raise ValueError("one target mapping is required per batch item")
        max_gt = max((int(target.get("boxes", torch.empty((0, 4))).reshape(-1, 4).shape[0]) for target in targets), default=0)
        gt_boxes = decoded.boxes.new_zeros((batch_size, max_gt, 4))
        gt_labels = torch.zeros((batch_size, max_gt, 1), dtype=torch.long, device=decoded.boxes.device)
        mask_gt = torch.zeros((batch_size, max_gt, 1), dtype=torch.bool, device=decoded.boxes.device)
        for index, target in enumerate(targets):
            boxes = target.get("boxes", gt_boxes.new_zeros((0, 4))).to(decoded.boxes.device, decoded.boxes.dtype).reshape(-1, 4)
            labels = target.get("labels", gt_labels.new_zeros((0,))).to(decoded.boxes.device, torch.long).reshape(-1)
            if boxes.shape[0] != labels.shape[0]:
                raise ValueError("target boxes and labels length mismatch")
            if boxes.shape[0]:
                gt_boxes[index, : boxes.shape[0]] = boxes
                gt_labels[index, : boxes.shape[0], 0] = labels
                mask_gt[index, : boxes.shape[0], 0] = True
        assignment = self.assigner(decoded.scores.detach(), decoded.boxes.detach(), decoded.points, gt_labels, gt_boxes, mask_gt)
        target_scores = assignment.target_scores
        pred_cls = raw_outputs["scores"].permute(0, 2, 1)
        pred_dist = raw_outputs["boxes"].permute(0, 2, 1).reshape(batch_size, anchors, 4, 16)
        denominator = target_scores.sum().clamp_min(1.0)
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, target_scores, reduction="none").sum() / denominator
        box_loss = decoded.boxes.sum() * 0.0
        dfl_loss = decoded.boxes.sum() * 0.0
        if assignment.foreground.any():
            positive = assignment.foreground
            weight = target_scores.sum(dim=-1)[positive]
            box_loss = (ciou_loss(decoded.boxes[positive], assignment.boxes[positive]) * weight).sum() / denominator
            points = decoded.points[None].expand(batch_size, -1, -1)[positive]
            strides = decoded.strides[None].expand(batch_size, -1)[positive]
            distances = torch.stack((points[:, 0] - assignment.boxes[positive, 0], points[:, 1] - assignment.boxes[positive, 1], assignment.boxes[positive, 2] - points[:, 0], assignment.boxes[positive, 3] - points[:, 1]), dim=-1) / strides[:, None]
            dfl = distribution_focal_loss(pred_dist[positive], distances)
            dfl_loss = (dfl * weight).sum() / denominator
        total = (self.cls_weight * cls_loss + self.box_weight * box_loss + self.dfl_weight * dfl_loss) * batch_size
        return {"loss": total, "cls_loss": cls_loss, "box_loss": box_loss, "dfl_loss": dfl_loss, "target_scores_sum": denominator.detach(), "positive_count": assignment.foreground.sum().to(total.dtype)}
