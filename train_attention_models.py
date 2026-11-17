"""Entry point for training and evaluating attention-memory students and their no-memory baseline."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import json
import math
import uuid

import torch

import path
from utils.runtime import run_lock, ATTENTION_MEMORY_NAMES, load_checkpoint, write_json, early_stopping_reached, seed_training
from utils.checkpoints import restore_rng
from routing.data import data_signature, guard_held_out, selected_dataset
from routing.predictions import sha256_file
from scripts.compare_vid import load_manifest
from models.frozen_efficientdet_teacher import EfficientDetTeacher
from models.attention_memory_detector import P4Student
from losses.multibox_loss import criterion_for
from engine import trainer_attention as trainer
# Preserve established imports while the implementations live in the engine.
from datasets.dataset_imagenetvid import TrainingPairs
from datasets.dataloader import collate_pairs
from engine.trainer_attention import evaluate, export_attention_run


def run(args):
    """Prepare or restore an experiment before invoking the engine."""
    selection, val = load_manifest(SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames,
                                                  manifest=args.manifest))
    manifest_hash = sha256_file(args.manifest)
    teacher_hash = sha256_file(args.teacher_checkpoint or path.DELTA_TEACHER_CHECKPOINT)
    teacher = EfficientDetTeacher(args.delta_root or path.DELTA_ROOT, args.teacher_checkpoint or path.DELTA_TEACHER_CHECKPOINT, args.device)
    latest = args.run_dir / 'detection.pt'
    saved = load_checkpoint(args.checkpoint) if args.checkpoint else None
    early_stopping = saved.get('early_stopping', False) if saved and args.early_stopping is None else bool(args.early_stopping)
    if saved and (saved.get('stage') != 'p4_student_v1' or saved['memory'] != args.memory or
                  saved.get('projection', True) is not False or
                  saved['teacher_sha256'] != teacher_hash or saved['label_to_name'] != val.label_to_name or
                  saved['category_id_to_label'] != val.category_id_to_label or
                  saved['manifest_sha256'] != manifest_hash):
        raise ValueError('Incompatible experiment/projection, teacher, categories or validation manifest.')
    defaults = dict(epochs=100, batch_size=4, learning_rate=.001, seed=0, validate_every=5000, max_batches=None)
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, saved['settings'][name] if saved else default)
    if (min(args.epochs, args.batch_size, args.validate_every) < 1 or
            (args.max_batches is not None and args.max_batches < 1) or
            args.num_workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid training budget, worker count or learning rate.')
    seed_training(args.seed)
    student = P4Student(args.memory, imagenet=not saved).to(args.device)
    if saved:
        student.load_state_dict(saved['model'], strict=True)
    if args.action == 'eval':
        if args.output_dir:
            metrics = trainer.export_attention_run(student, teacher, selection, val, args.device, args.output_dir, args.visuals)
            write_json(dict(checkpoint=str(args.checkpoint.resolve()), updates=saved['updates'], run_id=saved['run_id'],
                            teacher_sha256=teacher_hash, manifest_sha256=manifest_hash), args.output_dir/'evaluation_source.json')
        else:
            metrics = trainer.evaluate(student, teacher, selection, val, args.device)
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
    student_frames = saved['settings'].get('student_frames', 1) if saved else 6
    pairs = TrainingPairs(ds, student_frames)
    batches = min(len(pairs)//args.batch_size, args.max_batches or len(pairs))
    if batches < 1:
        raise ValueError('No complete batch of 10-frame training windows.')
    budget = batches * args.epochs
    settings = dict(train_sha256=sha256_file(args.train_annotations), subset_sha256=sha256_file(args.subset),
                    batch_size=args.batch_size, batches=batches, max_batches=args.max_batches, epochs=args.epochs, seed=args.seed,
                    learning_rate=args.learning_rate, validate_every=args.validate_every)
    if not saved or 'student_frames' in saved['settings']:
        settings['student_frames'] = student_frames
    if saved and saved['settings'] != settings:
        raise ValueError('Training data, sampling or cosine budget changed on resume.')
    optimizer = torch.optim.RMSprop(student.parameters(), lr=args.learning_rate, alpha=.9, momentum=.9, eps=1.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget)
    step, history = 0, []
    init_source = 'tinynet_d.in1k'
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
    if early_stopping_reached(saved, early_stopping, latest):
        return saved
    state = dict(stage='p4_student_v1', run_id=run_id, memory=args.memory, initialization=init_source,
                 projection=False, updates=step, history=history, settings=settings,
                 teacher_sha256=teacher_hash, manifest_sha256=manifest_hash,
                 label_to_name=val.label_to_name, category_id_to_label=val.category_id_to_label,
                 selection_metric='student_COCO_AP50')
    loader_options = SimpleNamespace(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
                                    prefetch_factor=args.prefetch_factor, sharing_strategy=args.sharing_strategy,
                                    device=args.device, task=args.task)
    return trainer.train(student, teacher, pairs, selection, val, optimizer, scheduler, criterion,
                         loader_options=loader_options, state=state, latest=latest, early_stopping=early_stopping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'eval'))
    parser.add_argument('--memory', required=True, choices=(*ATTENTION_MEMORY_NAMES.values(), *ATTENTION_MEMORY_NAMES),
                        help='no_memory or attention_memory; old none/p4 names are also accepted')
    parser.add_argument('--checkpoint', type=Path, help='Latest checkpoint for resume, or latest/best for evaluation')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, help='Fresh evaluation folder for predictions and metrics')
    parser.add_argument('--visuals', action='store_true', help='Also save annotated videos and frames')
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
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=None,
                        help='Stop after 10 validations without improvement; resume inherits the saved setting')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.memory = {v: k for k, v in ATTENTION_MEMORY_NAMES.items()}.get(args.memory, args.memory)
    if (args.output_dir or args.visuals) and args.action != 'eval':
        parser.error('Exports are evaluation-only.')
    if args.visuals and not args.output_dir:
        parser.error('--visuals requires --output-dir.')
    if args.action == 'eval' and not args.checkpoint:
        parser.error('Evaluation requires --checkpoint.')
    if args.run_dir is None:
        root = path.RUNS_DIR / 'p4_student_imagenet_no_projection'
        args.run_dir = root / ATTENTION_MEMORY_NAMES[args.memory]
        # Resume older server folders until they are renamed; never silently restart.
        if not args.run_dir.exists() and (root / args.memory).exists():
            args.run_dir = root / args.memory
    args.task, args.prefetch_factor, args.sharing_strategy = 'detection', 1, None
    if args.action == 'train':
        with run_lock(args.run_dir/'detection.pt'):
            run(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
