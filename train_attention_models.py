"""Entry point for training and evaluating attention-memory students and their no-memory baseline."""
import argparse
from pathlib import Path

import torch

import path
from utils.runtime import run_lock, ATTENTION_MEMORY_NAMES
# Preserve established imports while the implementations live in the engine.
from engine.attention import TrainingPairs, collate_pairs, evaluate, export_evaluation, run


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
