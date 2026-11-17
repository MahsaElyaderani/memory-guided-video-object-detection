"""Reusable LSTM training orchestration, detection loops, and streaming evaluation."""
from copy import copy
from pathlib import Path
import uuid
from collections import Counter
from dataclasses import dataclass
import time

import torch

from engine.evaluator import compute_map50, decode, export_lstm_run
from losses.multibox_loss import criterion_for
from models.joint_lstm_detector import corrupt_memory
from lstm_config import MemoryCorruption, PolicyConfig, probability
from utils.compute import profiled_costs, synchronize
from routing.data import data_signature
from utils.evaluation_data import load_frame, normalized_target, select_evaluation_frames
from routing.backbone_schedule import VideoContext, choose_big, forced_big, random_training_schedule
from routing.predictions import filtered
from utils.lstm_checkpoints import detector_fingerprint, lstm_metadata
from utils.checkpoints import capture_rng, restore_rng
from utils.resume import saved_arguments
from engine.classification_training import classification_epoch
from utils.runtime import tqdm, LOSS_KEYS, loss_postfix, early_stopping_reached
from datasets.dataloader import budget_loader

def train(model, train_data, val_data, optimizer, args, *, resume, metadata,
          sources, signatures, definition, early_stopping=False):
    """Train a prepared classification or detection run, validate, and save checkpoints."""
    from train_lstm_models import seed_training, save_matched_checkpoints

    destination = Path(args.checkpoint_out)
    history, run_id = (list(resume['history']), resume['run_id']) if resume else ([], uuid.uuid4().hex)

    def checkpoint(epoch, stats, budget, scheduler, validate):
        rng = capture_rng()
        was_training = model.training
        metrics = None
        if validate:
            try:
                seed_training(0)
                if args.task == 'classification':
                    val_args = copy(args)
                    val_args.max_batches = None
                    metrics = classification_epoch(model, val_data, val_args)
                else:
                    policy = PolicyConfig(mode='big' if model.standalone else 'fixed', interval=args.big_call_interval)
                    metrics, _ = evaluate(model, val_data, policy,
                        EvaluationOptions(args.device, score_threshold=0.,
                                          warmup_frames=0, progress=not args.no_progress))
            finally:
                restore_rng(rng)
                model.train(was_training)
        if validate:
            history.append({'epoch': epoch, 'train': stats, 'validation': metrics,
                            'update': budget['updates']})
        payload = {**lstm_metadata(model), 'matched_format_version': 1, 'stage': 'matched_' + args.task,
                   'experiment': args.experiment, 'initialization_seed': args.seed,
                   'protocol': {'bn_stats': 'frozen', 'standalone_memory': 'update', 'joint_memory': 'big_only'},
                   'class_to_idx': {str(i): i - 1 for i in range(1, 31)},
                   'category_id_to_label': metadata['category_id_to_label'], 'label_to_name': metadata['label_to_name'],
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng_state': capture_rng(),
                   'selection_metric': (resume.get('selection_metric', 'bce_loss' if args.task == 'classification' else 'mAP@0.5')
                                        if resume else 'micro_ap' if args.task == 'classification' else 'mAP@0.5'),
                   'epoch': epoch, 'run_id': run_id, 'history': history, 'sources': sources,
                   'training_signatures': signatures, 'run_definition': definition,
                   'resume_arguments': saved_arguments(args)}
        if args.task == 'classification' and model.backbone_only:
            payload['classification_architecture'] = 'backbone_only'
        payload.update(budget=budget, scheduler=scheduler, global_step=budget['updates'])
        if early_stopping:
            payload['early_stopping'] = True
        save_matched_checkpoints(payload, destination, select_best=validate)
        suffix = f" update {budget['updates']}"
        print(f'Saved {args.experiment} {args.task} epoch {epoch}{suffix}: {destination}', flush=True)
        if metrics:
            key = payload['selection_metric']
            print(f"Validation {key}={metrics[key]:.6f}; class coverage {len(metrics['present_classes'])}/30", flush=True)
        return payload

    return train_budgeted(model, train_data, args, optimizer, resume, checkpoint, early_stopping)


def train_budgeted(model, train_data, args, optimizer, resume, checkpoint, early_stopping=False):
    from train_lstm_models import seed_training

    if args.lr_schedule not in ('linear', 'cosine'):
        raise ValueError('Only linear and cosine training schedules are supported.')
    batches = min(len(train_data) // args.batch_size, args.max_batches or len(train_data))
    budget = {'epochs': args.epochs, 'total_updates': args.epochs * batches,
              'updates': 0, 'epoch': 1, 'batch': 0, 'sampling_version': 1}
    if resume:
        saved = resume.get('budget')
        if saved and 'origin_updates' in saved:
            if saved['batches_per_epoch'] != batches:
                raise ValueError('Continuation dataset/batch count changed on resume.')
            budget['total_updates'] = saved['total_updates']
        if not saved or any(saved[k] != budget[k] for k in ('epochs', 'total_updates', 'sampling_version')):
            raise ValueError('A budgeted schedule requires the original predetermined budget on resume.')
        budget.update(saved)
        if (not 0 <= budget['batch'] <= batches or
                budget['updates'] != budget.get('origin_updates', 0) +
                (budget['epoch'] - budget.get('origin_epoch', 1)) * batches + budget['batch'] or
                not 0 <= budget['updates'] <= budget['total_updates']):
            raise ValueError('Checkpoint batch position and completed update count disagree.')
        if args.learning_rate is not None and any(args.learning_rate != lr for lr in resume['scheduler']['base_lrs']):
            raise ValueError('Changing the initial LR would change the fixed-budget schedule; use a new run.')
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget['total_updates'])
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: max(0., 1. - step / budget['total_updates']))
    if resume:
        scheduler.load_state_dict(resume['scheduler'])
        if scheduler.last_epoch != budget['updates']:
            raise ValueError('Scheduler and completed update counts disagree.')
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = lr
    print(f"Budget: {budget['updates']}/{budget['total_updates']} updates; {args.lr_schedule} LR, no warmup.", flush=True)
    if budget['updates'] >= budget['total_updates']:
        return resume
    if resume and early_stopping_reached(resume, early_stopping, args.checkpoint_out):
        return resume
    # Step only after a successful optimizer update, including inside the reused loops.
    hook = optimizer.register_step_post_hook(lambda *_: scheduler.step())
    loop_args = copy(args)
    loop_args.max_batches = None  # The sliced loader already enforces the update budget.
    payload = resume
    try:
        while budget['updates'] < budget['total_updates']:
            if budget['batch'] == batches:
                budget['epoch'] += 1
                budget['batch'] = 0
            if budget['batch'] == 0:
                seed_training(args.seed + budget['epoch'])
            count = min(batches - budget['batch'], budget['total_updates'] - budget['updates'],
                        args.validate_every - budget['updates'] % args.validate_every)
            loader = budget_loader(train_data, args, budget['epoch'], budget['batch'], count)
            if args.task == 'classification':
                stats = classification_epoch(model, train_data, loop_args, optimizer, loader=loader)
            else:
                stats = train_batches(model, loader, optimizer, criterion_for(model), loop_args, budget['epoch'])
                if model.standalone:
                    stats.pop('big_selections')
            if scheduler.last_epoch != budget['updates'] + count:
                raise RuntimeError('Budgeted training must perform one optimizer update per consumed batch.')
            budget['updates'] += count
            budget['batch'] += count
            validate = budget['updates'] % args.validate_every == 0 or budget['updates'] == budget['total_updates']
            stats['interval_updates'] = count
            payload = checkpoint(budget['epoch'], stats, dict(budget), scheduler.state_dict(), validate)
            if validate and early_stopping_reached(payload, early_stopping):
                break
    finally:
        hook.remove()
    return payload


def train_batches(model, loader, optimizer, criterion, args, epoch=1):
    phase = "joint"
    model.configure_detector_training()
    totals, batches, selections, corrupted = dict.fromkeys(LOSS_KEYS, 0.), 0, 0, 0
    corruption = MemoryCorruption(args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std)
    description = f"Epoch {epoch} {phase} (current/mean)"
    with tqdm(loader, desc=description, disable=args.no_progress) as bar:
        for batch_index, (frames, targets) in enumerate(bar, 1):
            frames = frames.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            big_mask, small_commit = random_training_schedule(
                frames.shape[0], frames.shape[1], frames.device)
            state, logits, boxes = None, [], []
            for step in range(frames.shape[1]):
                output, state = model.forward_selected(frames[:, step], state, big_mask[:, step], small_commit[:, step])
                state, affected = corrupt_memory(state, big_mask[:, step], corruption)
                corrupted += int(affected.sum())
                logits.append(output["class_logits"])
                boxes.append(output["box_regression"])
            logits, boxes = torch.stack(logits, 1), torch.stack(boxes, 1)
            loss = criterion({"class_logits": logits, "box_regression": boxes,
                              "anchors": output["anchors"]}, targets)
            selections += int(big_mask.sum())
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

def evaluate(model, dataset, policy, options=None,
                    precision_ids=None, on_prediction=None):
    options = options or EvaluationOptions()
    precision_ids = set(precision_ids or [])
    device = torch.device(options.device)
    if policy.mode == "small" and precision_ids:
        raise ValueError("Small-only baseline cannot honor big-model precision requests.")
    select_evaluation_frames(dataset)
    if not precision_ids.issubset(dataset.routing_records):
        raise ValueError("Precision requests include unevaluated image IDs.")
    model.to(device).eval()
    costs = profiled_costs(model, device, options.warmup_frames)
    calls, reasons = Counter(big=0, small=0), Counter()
    predictions, targets, frame_ids, routes = [], [], [], []
    elapsed_total = 0.
    loss_totals = dict.fromkeys(LOSS_KEYS, 0.)
    criterion = None
    if options.compute_losses:
        criterion = criterion_for(model)
    with torch.no_grad(), tqdm(total=len(dataset.routing_records), desc=policy.mode, unit="frame",
                              disable=not options.progress) as bar:
        for video_id in dataset.video_ids:
            state, context = None, VideoContext()
            for step, record in enumerate(dataset.images_by_video[video_id]):
                target = normalized_target(dataset, record)
                synchronize(device)
                started = time.perf_counter()
                frame_cpu = load_frame(dataset, record, model.model_cfg.image_size, model.model_cfg.input_interpolation,
                                       paired=model.model_cfg.uses_timm)
                change = context.observe_frame(frame_cpu)
                scene = (policy.mode == "fixed" and policy.scene_change_threshold is not None
                         and change > policy.scene_change_threshold)
                if scene and policy.reset_on_scene_change:
                    state = None
                    context.scene_reset()
                previous_source, age = context.previous_big, context.contribution_age(step)
                frame = frame_cpu[None].to(device)
                state = model.initial_state(frame) if state is None else state
                forced = forced_big(policy, step, context.last_big_call, scene, int(record["id"]) in precision_ids)
                small_output, small_state = None, None
                if policy.mode == "small" or (not forced and policy.mode == "fixed"):
                    small_output, small_state = model.forward_small(frame, state)
                    calls["small"] += 1
                use_big, call_reasons = choose_big(policy, step, context.last_big_call, scene,
                                                   int(record["id"]) in precision_ids)
                if use_big:
                    output, new_state = model.forward_big(frame, state)
                    calls["big"] += 1
                else:
                    output, new_state = small_output, small_state
                if output is None:
                    raise RuntimeError("No selected detector output.")
                threshold = options.score_threshold
                previous_prediction = decode(output, model, threshold, options.max_detections)
                selected = filtered(previous_prediction, options.score_threshold, options.max_detections)
                combined = policy.mode == "fixed"
                contributed = use_big and (not combined or policy.memory_feedback != "none")
                if contributed:
                    state = new_state
                elif (policy.mode == "small" and policy.memory_feedback != "none"
                      and model.paired_config.small_only_state_policy == "update"):
                    state = new_state
                context.record(use_big, step, contributed)
                synchronize(device)
                elapsed = time.perf_counter() - started
                elapsed_total += elapsed
                current_losses = None
                if criterion is not None:
                    losses = criterion(output, [target])
                    current_losses = {key: float(losses[key]) for key in LOSS_KEYS}
                    if not all(torch.isfinite(torch.tensor(value)) for value in current_losses.values()):
                        raise FloatingPointError("Non-finite evaluation loss.")
                    for key in LOSS_KEYS:
                        loss_totals[key] += current_losses[key]
                prediction = {**selected, "image_id": target["image_id"]}
                predictions.append(prediction)
                targets.append({k: target[k] for k in ("boxes", "labels", "image_id")})
                frame_ids.append(int(record["id"]))
                reasons.update(call_reasons)
                route = {"image_id": int(record["id"]), "video_id": video_id,
                         "frame_id": int(record["frame_id"]), "video_step": step,
                         "selected_model": "big" if use_big else "small", "big_called": use_big,
                         "small_executed": small_output is not None, "reasons": call_reasons, "frame_change": change,
                         "scene_change": bool(scene), "previous_prediction_big": previous_source,
                         "previous_contribution_age": age, "memory_committed": contributed or
                         (policy.mode == "small" and policy.memory_feedback != "none"
                          and model.paired_config.small_only_state_policy == "update"),
                         "memory_corrupted": False, "inference_seconds": elapsed}
                routes.append(route)
                if on_prediction is not None:
                    on_prediction(record, target, prediction, route)
                bar.update(1)
                postfix = loss_postfix(current_losses, loss_totals, len(frame_ids)) if current_losses else {}
                postfix.update(big_calls=calls["big"], fps=f"{len(frame_ids)/elapsed_total:.2f}")
                bar.set_postfix(postfix)
    if not frame_ids:
        raise ValueError("No frames evaluated.")
    metrics = compute_map50(predictions, targets, num_classes=30, progress=options.progress)
    small_indices = [index for index, route in enumerate(routes) if route["selected_model"] == "small"]
    metrics["small_frame_image_ids"] = [frame_ids[index] for index in small_indices]
    metrics["small_frame_metrics"] = (compute_map50(
        [predictions[index] for index in small_indices], [targets[index] for index in small_indices],
        num_classes=30, progress=False) if small_indices else None)
    metrics["loss_computed"] = criterion is not None
    if criterion is not None:
        metrics.update({"mean_frame_" + key: value/len(frame_ids) for key,value in loss_totals.items()})
    counts = Counter(int(label) for target in targets for label in target["labels"])
    total_macs = sum(costs[name] * calls[name] for name in costs)
    metrics.update({"mode": policy.mode, "num_frames": len(frame_ids), "evaluated_image_ids": frame_ids,
                    "video_ids": list(dataset.video_ids), "frames_per_video": getattr(dataset, "paired_frames_per_video", None),
                    "class_names": dataset.label_to_name,
                    "ground_truth_counts_per_class": {i: counts[i] for i in range(1, 31)},
                    "score_threshold": options.score_threshold, "nms_iou_threshold": model.ssd_cfg.nms_iou_threshold,
                    "max_detections_per_class": options.max_detections, "max_total_detections": options.max_detections,
                    "big_call_frequency": calls["big"] / len(frame_ids), "executions": dict(calls),
                    "call_reasons": dict(reasons), "inference_seconds": elapsed_total,
                    "FPS": len(frame_ids) / elapsed_total, "device": str(device),
                    "conv_linear_MACs_per_branch": costs, "conv_linear_MACs_per_frame": total_macs / len(frame_ids),
                    "conv_linear_FLOPs_per_frame": 2 * total_macs / len(frame_ids),
                    "compute_scope": "Conv2d and Linear only; FLOPs=2*MACs. Excludes resize, BN, activations, pooling, recurrent elementwise operations, NMS and IO.",
                    "timing_scope": "Serial wall time including image decode/resize, transfer, scene check, model/router execution and postprocessing. Excludes warmup, metrics and visualization.",
                    "parameter_counts": {"total_detector": sum(p.numel() for p in model.parameters()),
                                         "big_backbone": sum(p.numel() for p in model.big.backbone.parameters()),
                                         "small_backbone": sum(p.numel() for p in model.fast_backbone.parameters()),
                                         "small_projection": sum(p.numel() for p in model.small_projection.parameters()),
                                         "shared_lstm": sum(p.numel() for p in model.big.lstm.parameters()),
                                         "shared_ssd": sum(p.numel() for p in model.big.feature_maps.parameters()) + sum(p.numel() for p in model.big.predictor.parameters())},
                    "detector_fingerprint": detector_fingerprint(model), "data_signature": data_signature(dataset),
                    "AP_definition": "Continuous precision-recall envelope AP at IoU 0.5; absent-GT classes omitted from mean."})
    return metrics, routes

@dataclass(frozen=True)
class EvaluationOptions:
    device: str = "cpu"
    score_threshold: float = 0.001
    max_detections: int = 100
    warmup_frames: int = 3
    seed: int = 0
    progress: bool = True
    compute_losses: bool = True

    def __post_init__(self):
        probability(self.score_threshold, "Evaluation score threshold")
        if self.max_detections < 1 or self.warmup_frames < 0:
            raise ValueError("Invalid detection limit or warmup count.")
