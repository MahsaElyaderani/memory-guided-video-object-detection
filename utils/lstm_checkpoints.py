"""Reusable LSTM checkpoint reconstruction, metadata, and fingerprints."""
from dataclasses import asdict
import hashlib
import json

import torch

from config import model_config_dict
from models.lstm_models import build_model


PAIRED_FORMAT_VERSION = 1


def paired_config_dict(model):
    config = asdict(model.paired_config)
    # Preserve fingerprints and resume definitions for pre-curriculum checkpoints.
    if config.get("fast_bn_momentum") is None:
        config.pop("fast_bn_momentum", None)
    return config


def detector_fingerprint(model):
    digest = hashlib.sha256()
    configuration = {"model": model_config_dict(model.model_cfg), "ssd": asdict(model.ssd_cfg),
                     "paired": paired_config_dict(model)}
    digest.update(json.dumps(configuration, sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lstm_metadata(model):
    return {"paired_format_version": PAIRED_FORMAT_VERSION,
            "model_config": model_config_dict(model.model_cfg), "ssd_config": asdict(model.ssd_cfg),
            "paired_config": paired_config_dict(model)}


def model_from_matched(payload):
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
