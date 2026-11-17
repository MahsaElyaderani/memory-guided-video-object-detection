"""Reusable LSTM checkpoint utilities for metadata and fingerprints."""
from dataclasses import asdict
import hashlib
import json

import torch

from config import model_config_dict


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
