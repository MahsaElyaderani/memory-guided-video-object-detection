"""Shared backbone configuration for standalone and joint LSTM models."""
from config import ModelConfig


def make_joint_config(backbone='timm_mobilenet_v2', width=1.):
    return ModelConfig(backbone=backbone, width_mult=width, image_size=320)
