"""Frame selection, loading, target normalization, and detection export for VID evaluation."""
import random

import torch
from PIL import Image

from datasets.transforms import resize_detection_views


def load_frame(dataset, record, image_size, interpolation="bilinear", paired=False):
    with Image.open(dataset.frames_root / record['file_name']) as image:
        if image.size != (int(record['width']), int(record['height'])):
            raise ValueError('Annotation dimensions differ from decoded small-model frame.')
        return resize_detection_views(image, image_size, interpolation, paired)


def normalized_target(dataset, record):
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
