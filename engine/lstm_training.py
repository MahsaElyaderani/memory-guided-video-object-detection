"""Reusable LSTM detection training engine."""
from dataclasses import asdict
import argparse
from types import SimpleNamespace
import math

import torch

from datasets.dataloader import build_dataloader
from datasets.transforms import TrainTransform
from engine.lstm_evaluation import select_evaluation_frames
# Preserve historical loss-helper imports through this engine.
from losses.multibox_loss import MultiBoxLoss, criterion_for
from models.joint_lstm_detector import corrupt_memory
from lstm_config import MemoryCorruption
from routing.data import data_signature, guard_held_out, selected_dataset, validate_native_categories
from routing.backbone_schedule import random_training_schedule
from utils.runtime import LOSS_KEYS, loss_postfix, tqdm


def detector_batch(model, frames, targets, criterion, corruption):
    big_mask, small_commit = random_training_schedule(
        frames.shape[0], frames.shape[1], frames.device)
    state, logits, boxes, corrupted = None, [], [], 0
    for step in range(frames.shape[1]):
        output, state = model.forward_selected(frames[:, step], state, big_mask[:, step], small_commit[:, step])
        state, affected = corrupt_memory(state, big_mask[:, step], corruption)
        corrupted += int(affected.sum())
        logits.append(output["class_logits"])
        boxes.append(output["box_regression"])
    logits, boxes = torch.stack(logits, 1), torch.stack(boxes, 1)
    loss = criterion({"class_logits": logits, "box_regression": boxes,
                      "anchors": output["anchors"]}, targets)
    return loss, {"big_selections": int(big_mask.sum()), "corrupted_updates": corrupted}


def train_detector_epoch(model, loader, optimizer, criterion, args, epoch=1):
    phase = "joint"
    model.configure_detector_training()
    totals, batches, selections, corrupted = dict.fromkeys(LOSS_KEYS, 0.), 0, 0, 0
    corruption = MemoryCorruption(args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std)
    description = f"Epoch {epoch} {phase} (current/mean)"
    with tqdm(loader, desc=description, disable=args.no_progress) as bar:
        for batch_index, (frames, targets) in enumerate(bar, 1):
            frames = frames.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, info = detector_batch(model, frames, targets, criterion, corruption)
            selections += info["big_selections"]
            corrupted += info["corrupted_updates"]
            if not torch.isfinite(loss["total_loss"]):
                raise FloatingPointError("Non-finite paired detector loss.")
            loss["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            current = {key: float(loss[key].detach()) for key in LOSS_KEYS}
            for key in LOSS_KEYS:
                totals[key] += current[key]
            batches += 1
            bar.set_postfix(loss_postfix(current, totals, batches))
            log_every = getattr(args, "log_every", 50)
            if batches == 1 or batches % log_every == 0:
                print(f"{description}, batch {batches}: " + ", ".join(
                    f"{key}={current[key]:.4f}/{totals[key]/batches:.4f}" for key in LOSS_KEYS), flush=True)
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
    if not batches:
        raise ValueError("No complete training batches.")
    print(f"Epoch {epoch} {phase} mean: " + ", ".join(
        f"{key}={totals[key]/batches:.4f}" for key in LOSS_KEYS), flush=True)
    return {**{key: value / batches for key, value in totals.items()}, "phase": phase, "optimizer_steps": batches,
            "big_selections": selections, "corrupted_updates": corrupted, "skipped_batches": 0,
            "big_shared_frozen": False}


def add_training_arguments(parser):
    from routing.data import add_data_arguments
    add_data_arguments(parser)
    parser.add_argument("--val-annotations")
    parser.add_argument("--val-frames-root")
    parser.add_argument("--val-video-ids")
    parser.add_argument("--max-val-videos", type=int)
    parser.add_argument("--val-frames-per-video", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=17)
    parser.add_argument("--clip-stride", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sharing-strategy", choices=("file_descriptor", "file_system"),
                        help="CPU tensor transfer strategy; file_system avoids per-storage cached file descriptors.")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-clip", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big-call-interval", type=int, default=8,
                        help="Call the big backbone every N frames during fixed-policy validation; not validation frequency.")
    parser.add_argument("--validation-interval", dest="big_call_interval", type=int,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print a persistent loss line every N detector batches, plus first batch and epoch mean.")


def build_training_data(args, model, metadata):
    if (min(args.epochs, args.batch_size, args.clip_length, args.clip_stride, args.prefetch_factor,
            args.big_call_interval) < 1 or args.num_workers < 0 or
            args.max_batches is not None and args.max_batches < 1 or
            not math.isfinite(args.grad_clip) or args.grad_clip <= 0):
        raise ValueError("Invalid training controls.")
    if bool(args.val_annotations) != bool(args.val_frames_root):
        raise ValueError("Supply both validation annotations and frames root.")
    print(f"Loading training annotations and indexing clips: {args.annotations}", flush=True)
    train = selected_dataset(args, model.model_cfg.image_size, args.clip_length, args.clip_stride, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.uses_timm)
    print(f"Training index ready: {len(train.video_ids)} videos, {len(train)} clips.", flush=True)
    train.transform = TrainTransform(model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation,
                                     paired=model.model_cfg.uses_timm)
    if not metadata:
        metadata.update(category_id_to_label=train.category_id_to_label, label_to_name=train.label_to_name,
                        ssd_config=asdict(model.ssd_cfg))
    validate_native_categories(metadata, train)
    if len(train) < args.batch_size:
        raise ValueError("Training selection has fewer complete clips than a full batch.")
    val = None
    if args.val_annotations:
        print(f"Loading validation annotations: {args.val_annotations}", flush=True)
        val_args = SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames_root,
                                   video_ids=args.val_video_ids, max_videos=args.max_val_videos, subset_seed=args.subset_seed)
        val = selected_dataset(val_args, model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.uses_timm)
        select_evaluation_frames(val, args.val_frames_per_video, args.subset_seed)
        validate_native_categories(metadata, val)
        guard_held_out([data_signature(train)], val)
    return train, val


def make_loader(dataset, args, shuffle=True, drop_last=True):
    return build_dataloader(dataset, args.batch_size, shuffle, args.num_workers,
                            pin_memory=torch.device(args.device).type == "cuda", drop_last=drop_last,
                            prefetch_factor=args.prefetch_factor,
                            sharing_strategy=getattr(args, "sharing_strategy", None))
