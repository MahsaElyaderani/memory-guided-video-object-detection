"""Reusable LSTM checkpoint utilities for loading, metadata, and fingerprints."""
from dataclasses import asdict
import hashlib
import json

import torch

from config import SSDConfig, model_config_dict, checkpoint_model_config
from models.lstm_ssd_detector import LSTMSSD
from models.joint_lstm_detector import JointLSTMDetector
from lstm_config import PairedConfig


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


def paired_from_payload(payload, expected_stage=None):
    if payload.get("paired_format_version") != PAIRED_FORMAT_VERSION:
        raise ValueError("Not a supported paired-model checkpoint.")
    if payload.get("stage") != "detector":
        raise ValueError("Unknown paired checkpoint stage.")
    if expected_stage is not None and payload["stage"] != expected_stage:
        raise ValueError(f"Expected paired {expected_stage} checkpoint.")
    model = JointLSTMDetector(LSTMSSD(checkpoint_model_config(payload["model_config"]),
                             SSDConfig(**payload["ssd_config"])),
                      PairedConfig(**payload["paired_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model
