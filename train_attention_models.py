"""Entry point for training and evaluating attention-memory students and their no-memory baseline."""
import argparse
import json
import random
import uuid
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

import path
from datasets.dataloader import budget_loader
from datasets.transforms import TrainTransform, resize_detection_views
from engine.evaluator import decode
from losses.multibox_loss import criterion_for
from models.frozen_efficientdet_teacher import EfficientDetTeacher
from models.attention_memory_detector import P4Student
from routing.data import data_signature, guard_held_out, selected_dataset, validate_native_categories
from routing.predictions import sha256_file
from scripts.compare_vid import load_manifest, score_predictions
from train_lstm_models import model_from_matched, save_matched_checkpoints, seed_training
from utils.checkpoints import capture_rng, restore_rng
from utils.runtime import run_lock, load_checkpoint, tqdm
from vid_categories import vid_to_dataset_labels


class TrainingPairs(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = TrainTransform(160, interpolation='bicubic')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video, start = self.dataset.clip_index[index]
        records = self.dataset.images_by_video[video]
        # Train on one of the nine student positions following a keyframe in a 10-frame window.
        gap = random.randint(1, 9)
        pair = [self.dataset._load_image_and_target(records[start + t]) for t in (0, gap)]
        images, targets = self.transform.augment(*zip(*pair))
        return images[0], resize_detection_views(images[1], 160, 'bicubic'), targets[1]


def collate_pairs(batch):
    keys, current, targets = zip(*batch)
    return list(keys), torch.stack(current), list(targets)


@torch.no_grad()
def evaluate(student, teacher, selection, dataset, device):
    student.eval()
    mapping = vid_to_dataset_labels(dataset)
    key_rows, small_rows, small_frames = [], [], []
    previous = None
    for frame in tqdm(selection['frames'], desc='Interval 10', unit='frame'):
        if previous != frame['video_id']:
            step, cache = 0, None
        with Image.open(dataset.frames_root / frame['file_name']) as source:
            image = source.convert('RGB')
        width, height = image.size
        if [width, height] != frame['original_size']:
            raise ValueError('Frame dimensions differ from validation annotations.')
        if step % 10 == 0:
            cache, detections = teacher.keyframe([image], detect=True)
            info = dict(id=frame['image_id'], width=width, height=height)
            for row in teacher.records(detections[0], info, 0.):
                row['category_id'] = mapping[row['category_id']]
                key_rows.append(row)
        else:
            small_frames.append(frame)
            view = resize_detection_views(image, 160, 'bicubic')[None].to(device)
            pred = decode(student(view, cache), student, 0.)
            for box, label, score in zip(pred['boxes'].tolist(), pred['labels'].tolist(), pred['scores'].tolist()):
                x, y, right, bottom = box
                small_rows.append(dict(image_id=frame['image_id'], category_id=label, score=score,
                                       bbox=[x*width, y*height, (right-x)*width, (bottom-y)*height]))
        previous, step = frame['video_id'], step + 1
    if not small_frames:
        raise ValueError('Validation needs at least one intervening student frame.')
    return {'student': score_predictions({**selection, 'frames': small_frames}, small_rows),
            'sequence': score_predictions(selection, key_rows + small_rows)}


def run(args):
    selection, val = load_manifest(SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames,
                                                  manifest=args.manifest))
    manifest_hash = sha256_file(args.manifest)
    teacher_hash = sha256_file(args.teacher_checkpoint or path.DELTA_TEACHER_CHECKPOINT)
    teacher = EfficientDetTeacher(args.delta_root or path.DELTA_ROOT, args.teacher_checkpoint or path.DELTA_TEACHER_CHECKPOINT, args.device)
    latest = args.run_dir / 'detection.pt'
    saved = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if saved and (saved.get('stage') != 'p4_student_v1' or saved['memory'] != args.memory or
                  saved['teacher_sha256'] != teacher_hash or saved['label_to_name'] != val.label_to_name or
                  saved['category_id_to_label'] != val.category_id_to_label or
                  saved['manifest_sha256'] != manifest_hash):
        raise ValueError('Incompatible experiment, teacher, categories or validation manifest.')
    defaults = dict(epochs=100, batch_size=4, learning_rate=.001, seed=0, validate_every=5000, max_batches=None)
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, saved['settings'][name] if saved else default)
    if (min(args.epochs, args.batch_size, args.validate_every) < 1 or
            (args.max_batches is not None and args.max_batches < 1) or
            args.num_workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid training budget, worker count or learning rate.')
    seed_training(args.seed)
    detector = None
    if args.init_detector:
        source = load_checkpoint(args.init_detector)
        if source.get('stage') != 'matched_detection' or source.get('experiment') not in (
                'small_tinynet_d', 'joint_efficientnet_b0_tinynet_d'):
            raise ValueError('Expected matched TinyNet-D standalone or EfficientNet/TinyNet-D joint detection checkpoint.')
        validate_native_categories(source, val)
        guard_held_out(source.get('training_signatures', []), val)
        detector = model_from_matched(source)
        del source
    student = P4Student(args.memory, imagenet=not (saved or detector), detector=detector).to(args.device)
    del detector
    if saved:
        student.load_state_dict(saved['model'], strict=True)
    if args.action == 'eval':
        metrics = evaluate(student, teacher, selection, val, args.device)
        print(json.dumps(metrics, indent=2))
        return metrics

    if saved and args.checkpoint.resolve() != latest.resolve():
        raise ValueError('Resume training from this experiment directory’s latest detection.pt.')
    if not saved and any(args.run_dir.glob('*.pt')):
        raise FileExistsError(f'Choose a fresh experiment directory or resume {latest}.')
    ds = selected_dataset(SimpleNamespace(annotations=args.train_annotations, frames_root=args.train_frames,
                          video_ids=args.subset, max_videos=None, subset_seed=args.seed), clip_length=10, clip_stride=10)
    if ds.label_to_name != val.label_to_name or ds.category_id_to_label != val.category_id_to_label:
        raise ValueError('Training and validation category mappings differ.')
    guard_held_out([data_signature(ds)], val)
    pairs = TrainingPairs(ds)
    batches = min(len(pairs)//args.batch_size, args.max_batches or len(pairs))
    if batches < 1:
        raise ValueError('No complete batch of 10-frame training windows.')
    budget = batches * args.epochs
    settings = dict(train_sha256=sha256_file(args.train_annotations), subset_sha256=sha256_file(args.subset),
                    batch_size=args.batch_size, batches=batches, max_batches=args.max_batches, epochs=args.epochs, seed=args.seed,
                    learning_rate=args.learning_rate, validate_every=args.validate_every)
    if saved and saved['settings'] != settings:
        raise ValueError('Training data, sampling or cosine budget changed on resume.')
    optimizer = torch.optim.RMSprop(student.parameters(), lr=args.learning_rate, alpha=.9, momentum=.9, eps=1.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
    step, history = 0, []
    init_source = sha256_file(args.init_detector) if args.init_detector else 'tinynet_d.in1k'
    run_id = saved['run_id'] if saved else uuid.uuid4().hex
    if saved:
        best_path = latest.with_name('detection_best.pt')
        if best_path.exists() and load_checkpoint(best_path).get('run_id') != run_id:
            raise FileExistsError('Best checkpoint belongs to a different run.')
        step, history, init_source = saved['updates'], saved['history'], saved['initialization']
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        if not 0 <= step <= budget or scheduler.last_epoch != step or scheduler.T_max != budget:
            raise ValueError('Optimizer update count differs from saved cosine schedule.')
    criterion = criterion_for(student)
    while step < budget:
        epoch, offset = divmod(step, batches)
        count = min(batches-offset, args.validate_every-step % args.validate_every)
        loader = budget_loader(pairs, args, epoch+1, offset, count, collate_fn=collate_pairs)
        student.train()
        for module in student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        loss_sum = 0.
        for keys, current, targets in tqdm(loader, desc=f'{args.memory} epoch {epoch+1}'):
            cache = teacher.keyframe(keys)[0] if args.memory == 'p4' else None
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(student(current.to(args.device), cache), targets)['total_loss']
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            loss_sum += loss.item()
        validate = step % args.validate_every == 0 or step == budget
        if validate:
            metrics = evaluate(student, teacher, selection, val, args.device)
            history.append(dict(epoch=epoch+1, update=step, train_loss=loss_sum/count,
                                validation={'student_COCO_AP50': metrics['student']['coco_AP50'], **metrics}))
        payload = dict(stage='p4_student_v1', run_id=run_id, memory=args.memory, initialization=init_source,
                       model=student.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       updates=step, epoch=epoch+1, rng=capture_rng(), history=history, settings=settings,
                       teacher_sha256=teacher_hash, manifest_sha256=manifest_hash,
                       label_to_name=val.label_to_name, category_id_to_label=val.category_id_to_label,
                       selection_metric='student_COCO_AP50')
        save_matched_checkpoints(payload, latest, select_best=validate)
        print(f'{step}/{budget} updates; loss={loss_sum/count:.4f}; saved {latest}', flush=True)
        if validate:
            print(f"Student COCO AP50={metrics['student']['coco_AP50']:.4f}; "
                  f"whole sequence={metrics['sequence']['coco_AP50']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'eval'))
    parser.add_argument('--memory', required=True, choices=('none', 'p4'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--init-detector', type=Path, help='Matched detection checkpoint supplying TinyNet-D and SSD weights')
    source.add_argument('--checkpoint', type=Path, help='Latest checkpoint for resume, or latest/best for evaluation')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--delta-root', type=Path)
    parser.add_argument('--teacher-checkpoint', type=Path)
    parser.add_argument('--subset', type=Path, default=path.MATCHED_TRAIN_SUBSET)
    parser.add_argument('--manifest', type=Path, default=path.P4_VALIDATION_MANIFEST)
    parser.add_argument('--train-annotations', type=Path, default=path.TRAIN_ANNOTATIONS)
    parser.add_argument('--train-frames', type=Path, default=path.TRAIN_FRAMES)
    parser.add_argument('--val-annotations', type=Path, default=path.VAL_ANNOTATIONS)
    parser.add_argument('--val-frames', type=Path, default=path.VAL_FRAMES)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-batches', type=int, help='Cap updates per epoch for a short run')
    parser.add_argument('--validate-every', type=int)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.action == 'eval' and not args.checkpoint:
        parser.error('Evaluation requires --checkpoint.')
    args.run_dir = args.run_dir or path.RUNS_DIR / 'p4_student' / args.memory
    args.task, args.prefetch_factor, args.sharing_strategy = 'detection', 1, None
    if args.action == 'train':
        with run_lock(args.run_dir/'detection.pt'):
            run(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
