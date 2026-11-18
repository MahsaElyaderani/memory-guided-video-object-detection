"""Checkpoint metadata, model reconstruction, RNG state, and training resume helpers."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random
import sys

import torch

from models.lstm_models import build_model
from utils.runtime import load_checkpoint, save_matched_checkpoints


# Random state for reproducible resume.
def capture_rng():
    """Capture Python and PyTorch random states, including available CUDA generators."""
    result = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    """Restore captured Python and PyTorch random states for reproducible continuation."""
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


# These describe the current invocation, not the saved training experiment.
LOCAL_OPTIONS = {
    "annotations", "frames_root", "val_annotations", "val_frames_root", "video_ids",
    "val_video_ids", "checkpoint_out", "resume", "resume_best_checkpoint", "checkpoint_in",
    "big_checkpoint", "fast_checkpoint", "detector_checkpoint", "backbone_weights",
    "epochs", "device", "num_workers", "prefetch_factor", "sharing_strategy",
    "no_progress", "log_every", "cudnn_benchmark", "allow_worker_count_change",
    "learning_rate", "fast_learning_rate", "backbone_init", "init_source", "random_backbones",
}


def resume_defaults(parser, args, argv=None, payload=None):
    """Only omitted training options inherit saved values; CLI overrides stay explicit."""
    if not args.resume:
        return args
    tokens = sys.argv[1:] if argv is None else argv
    explicit = set()
    for token in tokens:
        if token == "--":
            break
        if token.startswith("--"):
            parsed = parser._parse_optional(token)
            if parsed and parsed[0] is not None:
                explicit.add(parsed[0].dest)
    payload = load_checkpoint(args.resume) if payload is None else payload
    if payload.get('budget') and 'epochs' not in explicit:
        args.epochs = payload['budget']['epochs']
    saved = dict(payload.get("run_definition", {}).get("arguments", {}))
    saved.update(payload.get("resume_arguments", {}))
    if saved.get("freeze_big_in_joint", False):
        raise ValueError('Frozen-big training is archived; use the archived source to resume this checkpoint.')
    if "paired_config" in payload:
        saved.update(payload["paired_config"])
        saved["train_bn"] = not payload["paired_config"].get("freeze_bn_stats", True)
    for key, value in saved.items():
        if key == "validation_interval":
            key = "big_call_interval"
        if hasattr(args, key) and key not in LOCAL_OPTIONS and key not in explicit:
            setattr(args, key, value)
    return args


def saved_arguments(args):
    """Serialize training settings while retaining established checkpoint field names."""
    saved = {key: value for key, value in vars(args).items() if key not in LOCAL_OPTIONS}
    # Keep serialized run definitions compatible with pre-rename checkpoints.
    if "big_call_interval" in saved:
        saved["validation_interval"] = saved.pop("big_call_interval")
    return saved


def restore_optimizer(optimizer, state, learning_rate=None, fast_learning_rate=None):
    """Restore moments and group options before applying explicit LR overrides."""
    for value in (learning_rate, fast_learning_rate):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Learning rates must be finite and positive.")
    if fast_learning_rate is not None and len(state["param_groups"]) != 2:
        raise ValueError("A separate fast learning rate requires a saved two-group optimizer.")
    optimizer.load_state_dict(state)
    if learning_rate is not None:
        optimizer.param_groups[0]["lr"] = learning_rate
    if fast_learning_rate is not None:
        optimizer.param_groups[1]["lr"] = fast_learning_rate


# LSTM checkpoint reconstruction and training payloads.
PAIRED_FORMAT_VERSION = 1


def paired_config_dict(model):
    """Serialize LSTM settings without changing fingerprints for omitted optional BN momentum."""
    config = asdict(model.paired_config)
    # Preserve fingerprints and resume definitions for pre-curriculum checkpoints.
    if config.get("fast_bn_momentum") is None:
        config.pop("fast_bn_momentum", None)
    return config


def detector_fingerprint(model):
    """Hash architecture settings and ordered state tensors to identify an exact detector."""
    digest = hashlib.sha256()
    configuration = {"model": asdict(model.model_cfg), "ssd": asdict(model.ssd_cfg),
                     "paired": paired_config_dict(model)}
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lstm_metadata(model):
    """Serialize model, SSD, and LSTM settings using established checkpoint keys."""
    return {"paired_format_version": PAIRED_FORMAT_VERSION,
            "model_config": asdict(model.model_cfg), "ssd_config": asdict(model.ssd_cfg),
            "paired_config": paired_config_dict(model)}


def model_from_matched(payload):
    """Validate checkpoint architecture and protocol, rebuild the model, and load tensors strictly."""
    if payload.get('matched_format_version') != 1 or payload.get('stage') not in ('matched_classification', 'matched_detection'):
        raise ValueError('Expected a matched classification/detection checkpoint.')
    if payload.get('class_to_idx') != {str(i): i - 1 for i in range(1, 31)}:
        raise ValueError('Matched class mapping differs.')
    task = payload['stage'].removeprefix('matched_')
    architecture = payload.get('classification_architecture', 'recurrent')
    expected_architecture = ('backbone_only' if task == 'classification' and
                             not payload['experiment'].startswith('joint_') else 'recurrent')
    if architecture != expected_architecture:
        raise ValueError('Incompatible classification architecture metadata.')
    model = build_model(payload['experiment'], task, payload['initialization_seed'])
    expected = lstm_metadata(model)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Matched architecture/configuration differs.')
    if payload.get('protocol') != {'bn_stats': 'frozen', 'standalone_memory': 'update', 'joint_memory': 'big_only'}:
        raise ValueError('Matched training protocol differs.')
    model.load_state_dict(payload['model'], strict=True)
    return model


def save_training_checkpoint(model, optimizer, args, *, epoch, budget, scheduler,
                             validate, resume, metadata, sources, signatures, definition,
                             history, run_id, early_stopping=False):
    """Build the LSTM training payload and save latest/best without running validation."""
    destination = Path(args.checkpoint_out)
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
    return payload
