"""Reusable evaluation utilities for detection postprocessing, AP metrics, classification metrics, and prediction exports."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torchvision.ops import nms
from routing.predictions import detection

try:
    from utils.runtime import tqdm
except ImportError:
    from ..utils.runtime import tqdm

try:
    from models.anchors import BoxCoderScales, decode_boxes
except ImportError:
    from ..models.anchors import BoxCoderScales, decode_boxes


def decode(output, model, threshold, maximum=100):
    for key in ("class_logits", "box_regression"):
        if not torch.isfinite(output[key]).all():
            raise FloatingPointError("Non-finite detector output.")
    return detection(**postprocess_detections(
        output["class_logits"], output["box_regression"], output["anchors"], num_classes=30,
        score_threshold=threshold, nms_iou_threshold=model.ssd_cfg.nms_iou_threshold,
        max_detections_per_class=maximum, max_total_detections=maximum))


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (a1[:, None] + a2[None, :] - inter).clamp(min=1e-8)


def postprocess_detections(
    class_logits: Tensor,
    box_regression: Tensor,
    anchors: Tensor,
    num_classes: int = 30,
    score_threshold: float = 0.0,
    nms_iou_threshold: float = 0.5,
    max_detections_per_class: int = 100,
    max_total_detections: int = 100,
    box_coder_scales: BoxCoderScales = BoxCoderScales(),
) -> Dict[str, Tensor]:
    """Decode one frame of SSD predictions and apply class-wise NMS."""
    if class_logits.ndim == 3:
        if class_logits.shape[0] != 1:
            raise ValueError("Batched postprocess currently expects batch size 1.")
        class_logits = class_logits[0]
        box_regression = box_regression[0]
    if class_logits.ndim != 2 or box_regression.ndim != 2:
        raise ValueError("Expected class_logits [A,C] and box_regression [A,4].")
    if class_logits.shape[0] != anchors.shape[0] or box_regression.shape[0] != anchors.shape[0]:
        raise ValueError("Anchor and prediction counts must match.")
    if class_logits.shape[1] != num_classes + 1:
        raise ValueError("Expected background + num_classes logits.")

    boxes = decode_boxes(box_regression, anchors, scales=box_coder_scales, clip=True)
    scores_all = torch.sigmoid(class_logits)
    out_boxes: List[Tensor] = []
    out_scores: List[Tensor] = []
    out_labels: List[Tensor] = []

    for label in range(1, num_classes + 1):
        scores = scores_all[:, label]
        keep = scores > score_threshold
        if not keep.any():
            continue
        cls_boxes = boxes[keep]
        cls_scores = scores[keep]
        kept = nms(cls_boxes, cls_scores, nms_iou_threshold)[:max_detections_per_class]
        out_boxes.append(cls_boxes[kept])
        out_scores.append(cls_scores[kept])
        out_labels.append(torch.full((kept.numel(),), label, device=kept.device, dtype=torch.long))

    if not out_boxes:
        return {
            "boxes": boxes.new_zeros((0, 4)),
            "scores": boxes.new_zeros((0,)),
            "labels": torch.zeros((0,), device=boxes.device, dtype=torch.long),
        }
    boxes_out = torch.cat(out_boxes)
    scores_out = torch.cat(out_scores)
    labels_out = torch.cat(out_labels)
    order = torch.argsort(scores_out, descending=True)[:max_total_detections]
    return {"boxes": boxes_out[order], "scores": scores_out[order], "labels": labels_out[order]}


def _continuous_ap(recalls: Tensor, precisions: Tensor) -> float:
    """Area under the monotonic precision-recall envelope (VOC-style AP)."""
    mrec = torch.cat([recalls.new_tensor([0.0]), recalls, recalls.new_tensor([1.0])])
    mpre = torch.cat([precisions.new_tensor([0.0]), precisions, precisions.new_tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    change = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).flatten()
    return float(((mrec[change + 1] - mrec[change]) * mpre[change + 1]).sum().item())


def class_coverage(counts):
    counts = {i + 1: int(count) for i, count in enumerate(counts)}
    return {'positive_counts_per_class': counts,
            'present_classes': [i for i, count in counts.items() if count],
            'absent_classes': [i for i, count in counts.items() if not count]}


def classification_metrics(scores, targets):
    """Non-interpolated micro AP; equal scores share a single threshold."""
    if scores.shape != targets.shape or scores.ndim != 2 or not scores.numel():
        raise ValueError('Expected matching, nonempty [frames, classes] scores and targets.')
    if not torch.isfinite(scores).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('AP requires finite scores and binary targets.')
    order = scores.flatten().argsort(descending=True)
    ranked = scores.flatten()[order]
    positives = targets.flatten()[order].double().cumsum(0)
    ends = torch.cat((torch.where(ranked[:-1] != ranked[1:])[0], order.new_tensor([len(order)-1])))
    tp = positives[ends]
    precision = tp / (ends + 1)
    increments = torch.diff(tp, prepend=tp.new_zeros(1))
    ap = float((increments * precision).sum() / tp[-1]) if tp[-1] else 0.
    return {'micro_ap': ap, **class_coverage(targets.sum(0))}


def compute_map50(
    predictions: Sequence[Dict[str, Tensor]],
    ground_truths: Sequence[Dict[str, Tensor]],
    num_classes: int = 30,
    iou_threshold: float = 0.5,
    progress: bool = False,
) -> Dict[str, object]:
    """Compute per-class AP and mean AP at a fixed IoU threshold of 0.5.

    Classes with no ground-truth instances are omitted from the mean, which is
    important for small validation subsets. This implements standard detection
    AP matching and a continuous precision envelope; for publication-grade
    reproduction, results should still be cross-checked against the official
    ImageNet VID devkit.
    """
    gt_by_image_class: dict[tuple[int, int], Tensor] = {}
    gt_counts = [0] * (num_classes + 1)
    for gt in ground_truths:
        image_id = int(gt["image_id"].item()) if isinstance(gt["image_id"], Tensor) else int(gt["image_id"])
        boxes = gt["boxes"].detach().cpu()
        labels = gt["labels"].detach().cpu()
        for cls in range(1, num_classes + 1):
            cls_boxes = boxes[labels == cls]
            if cls_boxes.numel():
                gt_by_image_class[(image_id, cls)] = cls_boxes
                gt_counts[cls] += cls_boxes.shape[0]

    preds_by_class: dict[int, list[tuple[float, int, Tensor]]] = defaultdict(list)
    for pred in predictions:
        image_id = int(pred["image_id"].item()) if isinstance(pred["image_id"], Tensor) else int(pred["image_id"])
        boxes = pred["boxes"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        labels = pred["labels"].detach().cpu()
        for box, score, label in zip(boxes, scores, labels):
            cls = int(label.item())
            preds_by_class[cls].append((float(score.item()), image_id, box))

    ap_per_class: Dict[int, float] = {}
    for cls in tqdm(range(1, num_classes + 1), desc="AP classes",
                    unit="class", disable=not progress, dynamic_ncols=True):
        num_gt = gt_counts[cls]
        if num_gt == 0:
            continue
        entries = sorted(preds_by_class.get(cls, []), key=lambda x: x[0], reverse=True)
        matched: dict[int, Tensor] = {
            image_id: torch.zeros(boxes.shape[0], dtype=torch.bool)
            for (image_id, c), boxes in gt_by_image_class.items() if c == cls
        }
        tp = torch.zeros(len(entries), dtype=torch.float32)
        fp = torch.zeros(len(entries), dtype=torch.float32)
        for idx, (_, image_id, box) in enumerate(entries):
            gt_boxes = gt_by_image_class.get((image_id, cls))
            if gt_boxes is None or gt_boxes.numel() == 0:
                fp[idx] = 1.0
                continue
            ious = _box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= iou_threshold and not matched[image_id][best_idx]:
                tp[idx] = 1.0
                matched[image_id][best_idx] = True
            else:
                fp[idx] = 1.0
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / float(num_gt)
        precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-12)
        ap_per_class[cls] = _continuous_ap(recalls, precisions)

    map50 = float(sum(ap_per_class.values()) / len(ap_per_class)) if ap_per_class else 0.0
    return {"mAP@0.5": map50, "AP@0.5_per_class": ap_per_class, "evaluated_classes": len(ap_per_class),
            **class_coverage(gt_counts[1:])}


def export_lstm_run(model, dataset, policy, options, output_dir,
                      precision_ids=None, visual_options=None):
    """Evaluate a standalone or joint LSTM detector using the supplied backbone policy.

    Create a new output directory with per-frame predictions, a separate routing
    log, metrics, and run status. Optionally render detection videos and frames.
    Return the metrics; existing output directories are never overwritten.
    """
    import json
    from pathlib import Path
    from engine.trainer_lstm import evaluate
    from utils.evaluation_data import raw_detection_row
    from utils.runtime import write_json
    from dataclasses import asdict
    from engine.visualizer import DetectionVideoExporter
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json({"status": "running"}, output / "run_status.json")
    exporter = DetectionVideoExporter(dataset, output, policy.mode, **visual_options) if visual_options else None
    try:
        with (output / "detections.jsonl").open("x") as detections, (output / "routing.jsonl").open("x") as routing:
            def save(record, target, prediction, route):
                row = raw_detection_row(record, target, prediction, dataset.label_to_name)
                row["route"] = route
                detections.write(json.dumps(row, allow_nan=False) + "\n")
                routing.write(json.dumps(route, allow_nan=False) + "\n")
                if exporter:
                    exporter(record, target, prediction, route)
            metrics, _ = evaluate(model, dataset, policy, options, precision_ids, save)
        if exporter:
            exporter.close()
            write_json({"videos": exporter.video_index}, output / "videos_index.json")
        metrics.update(policy=asdict(policy))
        write_json(metrics, output / "metrics.json")
        write_json({"status": "complete", "num_frames": metrics["num_frames"]}, output / "run_status.json")
        return metrics
    except BaseException as error:
        if exporter:
            exporter.close()
        write_json({"status": "failed", "error": str(error)}, output / "run_status.json")
        raise


def export_attention_run(student, teacher, selection, dataset, device, output, visuals=False):
    """Evaluate a no-memory or attention-memory student with frozen teacher keyframes.

    Use the selected validation frames and the evaluator's 10-frame schedule.
    Save the selection, predictions with teacher/student routing, student-only
    and full-sequence metrics, and run status in a new output directory.
    Optionally render videos and frames. Return metrics without changing weights.
    """
    from collections import Counter
    import json
    from engine.trainer_attention import evaluate
    from engine.visualizer import DetectionVideoExporter
    from utils.evaluation_data import normalized_target, raw_detection_row
    from utils.runtime import write_json, ATTENTION_MEMORY_NAMES

    output.mkdir(parents=True, exist_ok=False)
    write_json({'status': 'running'}, output/'run_status.json')
    exporter = None
    try:
        if visuals:
            exporter = DetectionVideoExporter(dataset, output, ATTENTION_MEMORY_NAMES[student.memory], save_frames_every=1)
        write_json(selection, output/'selection.json')
        with (output/'detections.jsonl').open('x') as stream:
            def save(frame, rows, step):
                # Match score_predictions: discard degenerate boxes, retain top 100 per frame.
                rows = sorted((r for r in rows if r['bbox'][2] > 0 and r['bbox'][3] > 0),
                              key=lambda r: r['score'], reverse=True)[:100]
                width, height = frame['original_size']
                boxes = torch.tensor([r['bbox'] for r in rows], dtype=torch.float64).reshape(-1, 4)
                boxes[:, 2:] += boxes[:, :2]
                pred = dict(boxes=boxes/boxes.new_tensor([width, height, width, height]),
                            labels=torch.tensor([r['category_id'] for r in rows], dtype=torch.long),
                            scores=torch.tensor([r['score'] for r in rows], dtype=torch.float64))
                record = dataset.routing_records[frame['image_id']]
                target = normalized_target(dataset, record)
                route = dict(selected_model='teacher' if step % 10 == 0 else 'student_' + ATTENTION_MEMORY_NAMES[student.memory],
                             previous_contribution_age=step % 10)
                row = raw_detection_row(record, target, pred, dataset.label_to_name)
                row.update(route=route, coco_predictions=rows)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                if exporter:
                    exporter(record, target, pred, route)
            metrics = evaluate(student, teacher, selection, dataset, device, save)
        if exporter:
            exporter.close()
            write_json({'videos': exporter.video_index}, output/'videos_index.json')
        sequence = metrics['sequence']
        counts = dict(Counter(f['video_id'] for f in selection['frames']))
        write_json({**metrics, 'mAP@0.5': sequence['continuous_AP50'],
                    'AP@0.5_per_class': sequence['continuous_AP50_per_class'],
                    'num_frames': sequence['frames'], 'video_ids': list(counts), 'frames_per_video': counts,
                    'evaluated_image_ids': [f['image_id'] for f in selection['frames']],
                    'class_names': dataset.label_to_name,
                    'ground_truth_counts_per_class': sequence['positive_counts_per_class'],
                    'score_threshold': 0., 'max_detections_per_class': 100, 'max_total_detections': 100,
                    'nms_iou_threshold': {'student': student.ssd_cfg.nms_iou_threshold,
                                          'teacher': 'checkpoint configuration'}}, output/'metrics.json')
        write_json({'status': 'complete', 'num_frames': sequence['frames']}, output/'run_status.json')
        return metrics
    except BaseException as error:
        if exporter:
            exporter.close()
        write_json({'status': 'failed', 'error': str(error)}, output/'run_status.json')
        raise
