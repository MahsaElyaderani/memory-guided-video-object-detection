"""Shared training/evaluation data selection, integrity checks, and evaluation frame helpers."""
import json
import random
from pathlib import Path

import torch
from PIL import Image

from datasets.dataset_imagenetvid import DetectionDataset
from datasets.transforms import EvalTransform, resize_detection_views
from utils.runtime import select_video_subset
from utils.runtime import sha256_file


def load_frame(dataset, record, image_size, interpolation="bilinear", paired=False):
    """Load one RGB frame, check its dimensions, and prepare the configured input views."""
    with Image.open(dataset.frames_root / record['file_name']) as image:
        if image.size != (int(record['width']), int(record['height'])):
            raise ValueError('Annotation dimensions differ from decoded small-model frame.')
        return resize_detection_views(image, image_size, interpolation, paired)


def normalized_target(dataset, record):
    """Convert valid annotated boxes to normalized xyxy coordinates and dataset labels."""
    boxes, labels = [], []
    for ann in dataset.anns_by_image.get(int(record['id']), []):
        category = int(ann['category_id'])
        x, y, width, height = map(float, ann['bbox'])
        if category not in dataset.category_id_to_label or width <= 0 or height <= 0:
            continue
        boxes.append([x, y, x + width, y + height])
        labels.append(dataset.category_id_to_label[category])
    absolute_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    normalized_boxes = (absolute_boxes / absolute_boxes.new_tensor(
        [record['width'], record['height'], record['width'], record['height']])).clamp(0, 1)
    return {'boxes': normalized_boxes,
            'labels': torch.tensor(labels, dtype=torch.long), 'image_id': torch.tensor(int(record['id'])),
            'video_id': torch.tensor(int(record['video_id'])), 'frame_id': torch.tensor(int(record['frame_id']))}


def raw_detection_row(record, target, prediction, names):
    """Build an export record with predictions, ground truth, and original image dimensions."""
    width, height = int(record['width']), int(record['height'])
    return {'video_id': int(record['video_id']), 'frame_id': int(record['frame_id']), 'image_id': int(record['id']),
            'file_name': record['file_name'], 'original_size': [width, height],
            'predictions': [{'dataset_label': int(label), 'class_name': names[int(label)], 'score': float(score),
                              'box_xyxy_normalized': box,
                              'box_xyxy_pixels': [box[0]*width, box[1]*height, box[2]*width, box[3]*height]}
                             for box, label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(), prediction['scores'].tolist())],
            'ground_truth': [{'dataset_label': int(label), 'class_name': names[int(label)], 'box_xyxy_normalized': box}
                             for box, label in zip(target['boxes'].tolist(), target['labels'].tolist())]}


def select_evaluation_frames(dataset, frames_per_video=None, seed=0):
    """Optionally select consecutive frames, then refresh their image-ID lookup."""
    if frames_per_video is not None:
        if frames_per_video < 1 or dataset.clip_length != 1:
            raise ValueError("Positive frame cap and single-frame evaluation dataset required.")
        rng = random.Random(seed)
        for video_id in dataset.video_ids:
            records = dataset.images_by_video[video_id]
            # Match native training: short videos consume no RNG draw; only consecutive windows qualify.
            if len(records) < frames_per_video:
                continue
            starts = []
            for start in range(len(records) - frames_per_video + 1):
                window = records[start:start + frames_per_video]
                ids = [int(record['frame_id']) for record in window if 'frame_id' in record]
                if len(ids) == len(window) and all(b == a + 1 for a, b in zip(ids, ids[1:])):
                    starts.append(start)
            if not starts:
                raise ValueError(f"Video {video_id} has no {frames_per_video}-frame consecutive segment.")
            start = rng.choice(starts)
            dataset.images_by_video[video_id] = records[start:start + frames_per_video]
        dataset.clip_index = [(v, index) for v in dataset.video_ids
                              for index in range(len(dataset.images_by_video[v]))]
        dataset.paired_frames_per_video = frames_per_video
    dataset.routing_records = {int(r["id"]): r for v in dataset.video_ids for r in dataset.images_by_video[v]}
    return dataset


def add_data_arguments(parser):
    """Add annotation, frame-root, and reproducible video-selection CLI options."""
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--frames-root', required=True)
    parser.add_argument('--video-ids', help='JSON list or split manifest containing video_ids.')
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--subset-seed', type=int, default=7)


def selected_dataset(args, image_size=320, clip_length=1, clip_stride=1, interpolation="bilinear", paired=False):
    """Build a selected dataset and reject invalid IDs, manifests, dimensions, or frame gaps."""
    raw = json.loads(Path(args.annotations).read_text())
    for field in ('videos', 'images', 'categories'):
        ids = [int(row['id']) for row in raw[field]]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate IDs in {field}.')
    # Do not retain a second full annotation tree while constructing the dataset.
    del raw, ids
    dataset = DetectionDataset(args.annotations, args.frames_root, EvalTransform(image_size, interpolation=interpolation, paired=paired),
                                clip_length=clip_length, clip_stride=clip_stride)
    if args.video_ids:
        selection = json.loads(Path(args.video_ids).read_text())
        if isinstance(selection, dict):
            if selection.get('annotations_sha256') != sha256_file(args.annotations):
                raise ValueError('Video split manifest refers to different annotations.')
            selection = selection['video_ids']
        if not isinstance(selection, list) or not selection or len(set(selection)) != len(selection):
            raise ValueError('Video selection must be a non-empty list of distinct IDs.')
        chosen = set(map(int, selection))
        if not chosen.issubset(dataset.video_ids):
            raise ValueError('Video split includes missing videos.')
        dataset.video_ids = sorted(chosen)
        dataset.clip_index = [pair for pair in dataset.clip_index if pair[0] in chosen]
    select_video_subset(dataset, args.max_videos, args.subset_seed)
    for video_id in dataset.video_ids:
        records = dataset.images_by_video[video_id]
        if not records:
            raise ValueError('Selected video is empty.')
        for record in records:
            if not all(key in record for key in ('frame_id', 'width', 'height')):
                raise ValueError('Routing requires explicit frame_id, width and height for every image.')
            if int(record['width']) < 1 or int(record['height']) < 1:
                raise ValueError('Invalid annotation image dimensions.')
        indices = [int(record['frame_id']) for record in records]
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f'Video {video_id} contains frame gaps; provide consecutive video frames.')
    if clip_length > 1 and not dataset.clip_index:
        raise ValueError('No complete training clips for the requested length.')
    dataset.routing_records = {int(r['id']): r for v in dataset.video_ids for r in dataset.images_by_video[v]}
    return dataset


def data_signature(dataset):
    # Most frames share a video directory. Resolve that directory once rather than
    # walking every ancestor for every frame. Preserve individual file symlinks.
    """Record annotation identity, selected frames, and resolved directories for resume and overlap checks."""
    parents, directories = {}, set()
    for record in dataset.routing_records.values():
        path = dataset.frames_root / str(record['file_name'])
        if path.is_symlink():
            directories.add(str(path.resolve().parent))
        else:
            parent = path.parent
            if parent not in parents:
                parents[parent] = str(parent.resolve())
            directories.add(parents[parent])
    return {'annotations_sha256': sha256_file(dataset.annotation_file),
            'video_ids': list(dataset.video_ids), 'clip_length': dataset.clip_length,
            'clip_stride': dataset.clip_stride, 'image_ids': list(dataset.routing_records),
            'video_directories': sorted(directories)}


def validate_native_categories(payload, dataset):
    """Require checkpoint class mappings and box-coder scales to match evaluation data."""
    labels = {int(key): int(value) for key, value in payload['category_id_to_label'].items()}
    names = {int(key): value for key, value in payload['label_to_name'].items()}
    if labels != dataset.category_id_to_label or names != dataset.label_to_name or dataset.num_classes != 30:
        raise ValueError('Native checkpoint and dataset must have identical 30-class definitions.')
    scales = payload['ssd_config']
    if tuple(scales[key] for key in ('y_scale', 'x_scale', 'height_scale', 'width_scale')) != (10, 10, 5, 5):
        raise ValueError('This evaluator requires native box-coder scales 10/10/5/5.')


def guard_held_out(training_signatures, dataset):
    """Reject evaluation videos overlapping training IDs or resolved image directories."""
    current = data_signature(dataset)
    for trained in training_signatures:
        ids_overlap = (trained['annotations_sha256'] == current['annotations_sha256'] and
                       set(trained['video_ids']) & set(current['video_ids']))
        files_overlap = set(trained.get('video_directories', [])) & set(current['video_directories'])
        if ids_overlap or files_overlap:
            raise ValueError('Evaluation videos overlap joint training videos.')
