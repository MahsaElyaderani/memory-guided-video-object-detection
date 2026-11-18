"""Export fixed VID frames, then compare native detectors and the external Delta baseline.

Examples (from mobile_vod):
  python scripts/compare_vid.py export --manifest splits/val100x20_seed7.json
  python scripts/compare_vid.py compare --manifest splits/val100x20_seed7.json \
    --model joint_mobile=/path/detection_best.pt --delta-root ../delta-distillation-fork \
    --output results/vid_comparison --device cpu

FPS retains each evaluator's documented timing scope; it is not matched streaming latency.
"""
import argparse
from collections import defaultdict
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

# Set thread pools before importing numerical libraries, including in child processes.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from utils.path import VAL_ANNOTATIONS, VAL_FRAMES
from utils.metric import compute_map50
from utils.metric import export_lstm_run
from utils.evaluation_data import select_evaluation_frames
from evaluate_lstm_models import load_evaluation_model
from utils.config import LSTMPolicyConfig, LSTMEvalConfig
from utils.evaluation_data import selected_dataset, validate_native_categories, guard_held_out
from utils.runtime import sha256_file
from utils.evaluation_data import normalized_target, raw_detection_row


def write_json(path, value):
    """Write a new JSON report, rejecting existing files and non-finite numeric values."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def frame_manifest(dataset):
    """Export ordered frame identities, dimensions, class labels, and ground truth."""
    empty = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long), 'scores': torch.empty(0)}
    rows = []
    for record in dataset.routing_records.values():
        row = raw_detection_row(record, normalized_target(dataset, record), empty, dataset.label_to_name)
        del row['predictions']
        rows.append(row)
    return rows


def load_manifest(args):
    """Verify annotation identity and reconstruct exactly the frames and labels in the manifest."""
    selection = json.loads(args.manifest.read_text())
    if selection['annotations_sha256'] != sha256_file(args.annotations):
        raise ValueError('Manifest annotation hash differs; use the original annotation file.')
    dataset = selected_dataset(argparse.Namespace(annotations=args.annotations, frames_root=args.frames_root,
                               video_ids=None, max_videos=None, subset_seed=7))
    frames = selection['frames']
    ids = [f['image_id'] for f in frames]
    if not frames or len(ids) != len(set(ids)):
        raise ValueError('Manifest must contain nonempty, unique frame IDs.')
    grouped = defaultdict(list)
    for frame in frames:
        grouped[frame['video_id']].append(dataset.routing_records[frame['image_id']])
    dataset.video_ids = list(grouped)
    dataset.images_by_video = dict(grouped)
    select_evaluation_frames(dataset)
    for records in grouped.values():
        if any(b['frame_id'] != a['frame_id'] + 1 for a, b in zip(records, records[1:])):
            raise ValueError('Each manifest video must be one consecutive segment.')
    if frame_manifest(dataset) != frames or {str(k): v for k, v in dataset.label_to_name.items()} != selection['class_names']:
        raise ValueError('Manifest frames, ordering, labels or ground truth differ from annotations.')
    return selection, dataset


def score_predictions(selection, rows):
    """Score identical top-100 predictions with both existing continuous AP and COCO AP50."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    by_image = defaultdict(list)
    sizes = {f['image_id']: f['original_size'] for f in selection['frames']}
    for row in rows:
        if row['image_id'] not in sizes or str(row['category_id']) not in selection['class_names']:
            raise ValueError('Prediction contains an unknown image or category.')
        x, y, w, h = row['bbox']
        if not torch.isfinite(torch.tensor([x, y, w, h, row['score']])).all():
            raise ValueError('Non-finite prediction box or score.')
        if w <= 0 or h <= 0:
            continue  # Match the external baseline's removal of degenerate boxes.
        by_image[row['image_id']].append(row)
    targets, predictions, images, annotations, kept = [], [], [], [], []
    for frame in selection['frames']:
        image_id = frame['image_id']
        width, height = sizes[image_id]
        scale = torch.tensor([width, height, width, height])
        boxes = torch.tensor([g['box_xyxy_normalized'] for g in frame['ground_truth']]).reshape(-1, 4) * scale
        labels = [g['dataset_label'] for g in frame['ground_truth']]
        targets.append(dict(image_id=image_id, boxes=boxes, labels=torch.tensor(labels, dtype=torch.long)))
        images.append(dict(id=image_id, width=width, height=height))
        for box, label in zip(boxes.tolist(), labels):
            x, y, x2, y2 = box
            annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=label,
                                    bbox=[x, y, x2-x, y2-y], area=(x2-x)*(y2-y), iscrowd=0))
        selected = sorted(by_image[image_id], key=lambda r: r['score'], reverse=True)[:100]
        kept.extend(selected)
        boxes = torch.tensor([r['bbox'] for r in selected]).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        predictions.append(dict(image_id=image_id, boxes=boxes,
                                scores=torch.tensor([r['score'] for r in selected]),
                                labels=torch.tensor([r['category_id'] for r in selected], dtype=torch.long)))
    continuous = compute_map50(predictions, targets, 30)
    coco = COCO()
    coco.dataset = dict(images=images, annotations=annotations, categories=[
        dict(id=int(k), name=v) for k, v in selection['class_names'].items()], info={})
    coco.createIndex()
    detections = coco.loadRes(kept) if kept else COCO()
    if not kept:
        detections.dataset = {**coco.dataset, 'annotations': []}
        detections.createIndex()
    evaluator = COCOeval(coco, detections, 'bbox')
    evaluator.params.imgIds = list(sizes)  # Never drop a frame with zero detections.
    evaluator.params.iouThrs = np.array([.5])
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval['precision'][0, :, :, 0, -1]
    per_class = {int(label): float(precision[:, i][precision[:, i] >= 0].mean())
                 for i, label in enumerate(evaluator.params.catIds) if (precision[:, i] >= 0).any()}
    return {'coco_AP50': sum(per_class.values()) / len(per_class) if per_class else 0.,
            'coco_AP50_per_class': per_class, 'continuous_AP50': continuous['mAP@0.5'],
            'continuous_AP50_per_class': continuous['AP@0.5_per_class'],
            'present_classes': continuous['present_classes'], 'absent_classes': continuous['absent_classes'],
            'positive_counts_per_class': continuous['positive_counts_per_class'], 'frames': len(sizes)}


def compare(args):
    """Evaluate selected native and external Delta models on one verified frame manifest."""
    import pycocotools  # Fail before inference if the official evaluator is unavailable.
    selection, dataset = load_manifest(args)
    models = dict(item.split('=', 1) for item in args.model)
    if len(models) != len(args.model) or any(not name.isidentifier() or name.startswith('delta_') for name in models):
        raise ValueError('Use unique simple model names: --model NAME=CHECKPOINT.')
    if not models and not args.delta_root:
        raise ValueError('Provide --model and/or --delta-root.')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'selection.json', selection)
    results = {}
    for name, checkpoint in models.items():
        checkpoint_hash = sha256_file(checkpoint)
        model, payload = load_evaluation_model(checkpoint)
        if payload.get('validation_pending'):
            raise ValueError('Checkpoint validation is pending.')
        validate_native_categories(payload, dataset)
        guard_held_out(payload.get('training_signatures', []), dataset)
        mode = 'big' if getattr(model, 'standalone', None) else 'fixed'
        metrics = export_lstm_run(model, dataset, LSTMPolicyConfig(mode, 10),
                                   LSTMEvalConfig(args.device, 0., 100, 10, 0, True, False),
                                   args.output / name, visual_options={} if args.visuals else None)
        rows = []
        for line in (args.output / name / 'detections.jsonl').read_text().splitlines():
            frame = json.loads(line)
            for pred in frame['predictions']:
                x, y, x2, y2 = pred['box_xyxy_pixels']
                rows.append(dict(image_id=frame['image_id'], category_id=pred['dataset_label'],
                                 bbox=[x, y, x2-x, y2-y], score=pred['score']))
        if sha256_file(checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation; use a stable copy.')
        results[name] = {**score_predictions(selection, rows), 'FPS': metrics['FPS'],
                         'timing_scope': metrics['timing_scope'], 'execution': 'sequential frames',
                         'partial_GFLOPs': metrics['conv_linear_FLOPs_per_frame']/1e9,
                         'compute_scope': metrics['compute_scope'], 'checkpoint_sha256': checkpoint_hash}
        del model, payload
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if args.delta_root:
        baseline = args.delta_root.resolve() / 'baseline'
        reference = json.loads((baseline / 'selection.json').read_text())
        if reference['class_names'] != selection['class_names']:
            raise ValueError('Delta baseline class mapping differs from the manifest.')
        checkpoint = args.delta_checkpoint or args.delta_root / 'checkpoints/test/efficientdet_d0.pth'
        command = [sys.executable, str(baseline / 'evaluate_tracking.py'), '--deltadistill-path', str(args.delta_root.resolve()),
                   '--frames', str(args.frames_root), '--selection', str(args.manifest.resolve()),
                   '--checkpoint', str(checkpoint), '--output', str(args.output / 'delta_baselines'),
                   '--device', args.device, '--threads', '1', '--score-threshold', '0']
        checkpoint_hash = sha256_file(checkpoint)
        subprocess.run(command, check=True)
        if sha256_file(checkpoint) != checkpoint_hash:
            raise ValueError('Delta checkpoint changed during evaluation.')
        folder = args.output / 'delta_baselines'
        if json.loads((folder / 'image_ids.json').read_text()) != [f['image_id'] for f in selection['frames']]:
            raise ValueError('Delta evaluated different frames or frame ordering.')
        summary = json.loads((folder / 'summary.json').read_text())
        for name, metrics in summary['results'].items():
            results['delta_' + name] = {**score_predictions(selection, json.loads((folder / (name+'.json')).read_text())),
                'FPS': metrics['FPS'], 'timing_scope': summary['timing'],
                'execution': '10-frame batching' if name == 'delta' else 'sequential frames / tracking',
                'partial_GFLOPs': metrics['neural_conv_GFLOPs_per_frame'], 'compute_scope': summary['flops_scope'],
                'checkpoint_sha256': checkpoint_hash}
        if args.visuals:
            subprocess.run([sys.executable, str(baseline/'visualize.py'), '--results', str(folder),
                            '--selection', str(args.manifest), '--frames', str(args.frames_root)], check=True)
    report = dict(interval=10, device=args.device, threads=1, dtype='float32',
                  manifest_sha256=sha256_file(args.manifest), results=results, torch_version=str(torch.__version__),
                  hardware=torch.cuda.get_device_name() if args.device == 'cuda' else platform.processor(),
                  note='FPS timing scopes and partial FLOP scopes differ; not a matched latency/total-compute ranking.')
    write_json(args.output / 'comparison.json', report)
    lines = ['| Model | COCO AP50 | Continuous AP50 | FPS* | Partial GFLOPs* |',
             '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        lines.append(f"| {name} | {result['coco_AP50']:.2%} | {result['continuous_AP50']:.2%} | "
                     f"{result['FPS']:.2f} | {result['partial_GFLOPs']:.3f} |")
    lines += ['', '*FPS and operation-count scopes differ; see comparison.json. Paper-only results are not measured here.']
    (args.output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    """Dispatch manifest export or fixed-manifest model comparison from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('export', 'compare'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=VAL_ANNOTATIONS)
    parser.add_argument('--frames-root', type=Path, default=VAL_FRAMES)
    parser.add_argument('--model', action='append', default=[], metavar='NAME=CHECKPOINT')
    parser.add_argument('--delta-root', type=Path)
    parser.add_argument('--delta-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--visuals', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import cv2
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    if args.action == 'compare':
        if args.output is None:
            parser.error('--output is required for compare (must be a new directory).')
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA is unavailable; choose --device cpu.')
        compare(args)
    else:
        dataset = selected_dataset(argparse.Namespace(annotations=args.annotations, frames_root=args.frames_root,
                                   video_ids=None, max_videos=100, subset_seed=7))
        select_evaluation_frames(dataset, 20, 7)
        frames = frame_manifest(dataset)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.manifest, dict(format_version=1, annotations_sha256=sha256_file(args.annotations),
                   class_names=dataset.label_to_name, num_frames=len(frames), num_videos=len(dataset.video_ids),
                   subset_seed=7, frames_per_video=20, frames=frames))
        print(f'Exported {len(dataset.video_ids)} videos / {len(frames)} frames to {args.manifest}')


if __name__ == '__main__':
    main()
