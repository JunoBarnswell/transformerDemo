from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .coordinates import box_iou
from .types import Detection


def _average_precision(scores: list[float], true_positives: list[int], false_positives: list[int], num_ground_truth: int) -> float:
    if num_ground_truth <= 0:
        return 0.0
    if not scores:
        return 0.0
    tp = torch.tensor(true_positives, dtype=torch.float64).cumsum(0)
    fp = torch.tensor(false_positives, dtype=torch.float64).cumsum(0)
    recall = tp / float(num_ground_truth)
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall_points = torch.cat((torch.zeros(1, dtype=recall.dtype), recall, torch.ones(1, dtype=recall.dtype)))
    precision_points = torch.cat((torch.ones(1, dtype=precision.dtype), precision, torch.zeros(1, dtype=precision.dtype)))
    for index in range(precision_points.numel() - 2, -1, -1):
        precision_points[index] = torch.maximum(precision_points[index], precision_points[index + 1])
    grid = torch.linspace(0, 1, 101, dtype=recall.dtype)
    interpolated = torch.zeros_like(grid)
    for index, value in enumerate(grid):
        eligible = torch.where(recall_points >= value)[0]
        interpolated[index] = precision_points[eligible[0]] if eligible.numel() else precision_points[-1]
    return float(torch.trapezoid(interpolated, grid).item())


def detection_ap(
    predictions: Sequence[Sequence[Detection]],
    targets: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
    num_classes: int,
) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of images")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0,1]")
    per_class: dict[int, float] = {}
    all_tp = 0
    all_fp = 0
    total_gt = 0
    for class_id in range(num_classes):
        class_predictions = []
        ground_truth_by_image: list[torch.Tensor] = []
        class_gt = 0
        for image_index, (image_predictions, target) in enumerate(zip(predictions, targets)):
            target_boxes = target.get("boxes", torch.zeros((0, 4)))
            target_labels = target.get("labels", torch.zeros((0,), dtype=torch.long))
            target_boxes = target_boxes.detach().cpu().float().reshape(-1, 4)
            target_labels = target_labels.detach().cpu().long().reshape(-1)
            gt = target_boxes[target_labels == class_id]
            ground_truth_by_image.append(gt)
            class_gt += gt.shape[0]
            for prediction in image_predictions:
                if prediction.class_id == class_id:
                    class_predictions.append((float(prediction.score), image_index, prediction.box_xyxy))
        class_predictions.sort(key=lambda item: item[0], reverse=True)
        matched = [torch.zeros((gt.shape[0],), dtype=torch.bool) for gt in ground_truth_by_image]
        tp_values: list[int] = []
        fp_values: list[int] = []
        for _, image_index, box in class_predictions:
            predicted_box = torch.tensor([box], dtype=torch.float32)
            gt = ground_truth_by_image[image_index]
            if gt.numel() == 0:
                tp_values.append(0)
                fp_values.append(1)
                continue
            overlaps = box_iou(predicted_box, gt)[0]
            best_iou, best_index = overlaps.max(dim=0)
            if best_iou >= iou_threshold and not matched[image_index][best_index]:
                matched[image_index][best_index] = True
                tp_values.append(1)
                fp_values.append(0)
            else:
                tp_values.append(0)
                fp_values.append(1)
        per_class[class_id] = _average_precision(
            [item[0] for item in class_predictions], tp_values, fp_values, class_gt
        )
        all_tp += sum(tp_values)
        all_fp += sum(fp_values)
        total_gt += class_gt
    return {
        "ap": sum(per_class.values()) / max(num_classes, 1),
        "precision": all_tp / max(all_tp + all_fp, 1),
        "recall": all_tp / max(total_gt, 1),
        "false_positives_per_image": all_fp / max(len(predictions), 1),
        "per_class_ap": {str(key): value for key, value in per_class.items()},
        "evaluated_classes": [int(key) for key in per_class],
    }


def evaluate_detection(
    predictions: Sequence[Sequence[Detection]],
    targets: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
) -> dict[str, Any]:
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    by_threshold = {threshold: detection_ap(predictions, targets, iou_threshold=threshold, num_classes=num_classes) for threshold in thresholds}
    small_gt = 0
    small_tp = 0
    for image_predictions, target in zip(predictions, targets):
        target_boxes = target.get("boxes", torch.zeros((0, 4))).detach().cpu().float().reshape(-1, 4)
        target_labels = target.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu().long().reshape(-1)
        small = ((target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])) / (640.0 * 640.0) <= 0.01
        small_gt += int(small.sum())
        used: set[int] = set()
        for prediction in sorted(image_predictions, key=lambda item: item.score, reverse=True):
            for index in torch.where(small & (target_labels == prediction.class_id))[0].tolist():
                if index in used:
                    continue
                overlap = box_iou(torch.tensor([prediction.box_xyxy]), target_boxes[index:index + 1])[0, 0]
                if overlap >= 0.5:
                    used.add(index)
                    small_tp += 1
                    break
    result = {
        "map50": by_threshold[0.5]["ap"],
        "map75": by_threshold[0.75]["ap"],
        "map50_95": sum(item["ap"] for item in by_threshold.values()) / len(by_threshold),
        "precision": by_threshold[0.5]["precision"],
        "recall": by_threshold[0.5]["recall"],
        "false_positives_per_image": by_threshold[0.5]["false_positives_per_image"],
        "per_class_ap50": by_threshold[0.5]["per_class_ap"],
        "evaluated_classes": by_threshold[0.5]["evaluated_classes"],
        "small_recall": small_tp / max(small_gt, 1),
        "small_ground_truth": small_gt,
    }
    return result


def calibrate_detection_threshold(
    predictions: Sequence[Sequence[Detection]],
    targets: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
    target_recall: float = 0.95,
) -> dict[str, float | None]:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0,1]")
    candidates = sorted({0.001, *[round(float(item.score), 6) for row in predictions for item in row]}, reverse=True)
    best_f1 = (-1.0, None)
    best_target = None
    for threshold in candidates:
        filtered = [[item for item in row if item.score >= threshold] for row in predictions]
        measured = detection_ap(filtered, targets, iou_threshold=0.5, num_classes=num_classes)
        precision, recall = float(measured["precision"]), float(measured["recall"])
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1[0]:
            best_f1 = (f1, threshold)
        if recall >= target_recall:
            best_target = threshold
    return {"best_f1_threshold": best_f1[1], "target_recall_threshold": best_target}


def evaluate_segmentation(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    *,
    num_classes: int,
) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same number of images")
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)
    predicted_count = torch.zeros(num_classes, dtype=torch.float64)
    target_count = torch.zeros(num_classes, dtype=torch.float64)
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape:
            raise ValueError("segmentation prediction and target shapes must match")
        prediction = prediction.detach().cpu().long()
        target = target.detach().cpu().long()
        for class_id in range(num_classes):
            predicted = prediction == class_id
            expected = target == class_id
            intersection[class_id] += (predicted & expected).sum()
            union[class_id] += (predicted | expected).sum()
            predicted_count[class_id] += predicted.sum()
            target_count[class_id] += expected.sum()
    iou = intersection / union.clamp_min(1.0)
    dice = 2.0 * intersection / (predicted_count + target_count).clamp_min(1.0)
    foreground = torch.arange(num_classes) > 0
    defect_tp = intersection[foreground].sum()
    defect_precision = defect_tp / predicted_count[foreground].sum().clamp_min(1.0)
    defect_recall = defect_tp / target_count[foreground].sum().clamp_min(1.0)
    return {
        "miou": float(iou.mean().item()),
        "dice": float(dice.mean().item()),
        "defect_precision": float(defect_precision.item()),
        "defect_recall": float(defect_recall.item()),
        "per_class_iou": {str(index): float(value) for index, value in enumerate(iou.tolist())},
    }
