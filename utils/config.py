"""Model, SSD, and LSTM policy settings; training options live in entry points and launchers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Backbone, shared Bottleneck-LSTM, and SSD architecture settings."""

    image_size: int = 320
    width_mult: float = 1.0
    mobilenet_output_channels: int = 1280
    lstm_channels: int = 256
    lstm_kernel_size: int = 3
    lstm_forget_bias: float = 1.0
    sequence_length: int = 10
    num_classes: int = 30  # foreground ImageNet-VID classes

    # Detection feature maps in Table 1:
    # Bottleneck-LSTM output, FM1, FM2, FM3, FM4.
    feature_channels: Tuple[int, ...] = (256, 256, 128, 128, 64)

    # The paper equations use ReLU. The later released TensorFlow code uses
    # ReLU6 when constructing the cell; choose 'relu' for paper fidelity.
    lstm_activation: str = "relu"

    backbone: str = "timm_mobilenet_v2"

    @property
    def uses_timm(self):
        """Identify backbones using the timm input and feature conventions."""
        return self.backbone in ('timm_mobilenet_v2', 'tinynet_d', 'efficientnet_b0')

    @property
    def input_interpolation(self):
        """Return the resize interpolation required by the configured backbone family."""
        return "bicubic" if self.uses_timm else "bilinear"

    def __post_init__(self) -> None:
        """Reject unsupported backbone widths, image sizes, and LSTM or SSD dimensions."""
        if not self.uses_timm:
            raise ValueError("Unknown or archived backbone family.")
        widths = (0.5, 1.0, 1.1, 1.2) if self.backbone == "timm_mobilenet_v2" else (1.0,)
        if self.width_mult not in widths:
            raise ValueError(f"Supported {self.backbone} widths are {widths}.")
        if self.image_size not in (160, 320):
            raise ValueError("Inputs must be 320 (big) or 160 (small).")
        if self.mobilenet_output_channels != 1280:
            raise ValueError("Backbone output channels must be 1280 for this width.")
        if self.lstm_channels != 256:
            raise ValueError("Paper alpha=1 Bottleneck-LSTM output/state is 256 channels.")
        if len(self.feature_channels) != 5:
            raise ValueError("The paper architecture uses five detection feature levels.")
        if self.feature_channels[0] != self.lstm_channels:
            raise ValueError("First detection feature is the Bottleneck-LSTM output.")
        if self.lstm_activation not in {"relu", "relu6"}:
            raise ValueError("lstm_activation must be 'relu' or 'relu6'.")


@dataclass(frozen=True)
class SSDConfig:
    """SSD anchor/coder settings.

    The anchor settings and box-coder scales are taken from the authors'
    released ImageNet-VID TensorFlow configuration. TensorFlow's SSD anchor
    generator also adds an interpolated-scale square anchor on levels > 0 and,
    by default, uses only three boxes on the finest level.
    """

    num_levels: int = 5
    min_scale: float = 0.20
    max_scale: float = 0.95
    aspect_ratios: Tuple[float, ...] = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)
    interpolated_scale_aspect_ratio: float = 1.0
    reduce_boxes_in_lowest_layer: bool = True

    match_iou_threshold: float = 0.50
    nms_iou_threshold: float = 0.50

    # TensorFlow FasterRcnnBoxCoder settings in the released config.
    y_scale: float = 10.0
    x_scale: float = 10.0
    height_scale: float = 5.0
    width_scale: float = 5.0

    # The predictor outputs one background logit plus the 30 foreground logits.
    include_background_class: bool = True

    @property
    def num_anchors_per_location(self) -> Tuple[int, ...]:
        # TensorFlow SSD default: finest level has 3 boxes when
        # reduce_boxes_in_lowest_layer=True. Other levels contain one anchor
        # per configured aspect ratio plus the interpolated-scale square box.
        """Compute anchor counts for the finest level and remaining SSD feature levels."""
        first = 3 if self.reduce_boxes_in_lowest_layer else (
            len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        )
        other = len(self.aspect_ratios) + int(self.interpolated_scale_aspect_ratio > 0)
        return (first,) + (other,) * (self.num_levels - 1)


def probability(value, name):
    """Reject non-finite values or values outside the closed unit interval."""
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1].")


@dataclass(frozen=True)
class LSTMConfig:
    """Small-branch dimensions, batch-normalization behavior, and LSTM memory policy."""
    fast_width_mult: float = 0.5
    fast_image_size: int = 160
    min_depth: int = 8
    small_only_state_policy: str = "update"
    freeze_bn_stats: bool = True
    fast_bn_momentum: float | None = None

    def __post_init__(self):
        """Validate small-branch dimensions, memory policy, and optional BN momentum."""
        if not math.isfinite(self.fast_width_mult) or not 0 < self.fast_width_mult <= 1:
            raise ValueError("fast_width_mult must be in (0,1].")
        if self.fast_image_size < 16 or self.fast_image_size % 16 or self.min_depth < 1:
            raise ValueError("Fast input must be a positive multiple of 16; min_depth must be positive.")
        if self.small_only_state_policy not in ("update", "hold"):
            raise ValueError("small_only_state_policy must be update or hold.")
        if self.fast_bn_momentum is not None:
            if not math.isfinite(self.fast_bn_momentum) or not 0 < self.fast_bn_momentum <= 1:
                raise ValueError("fast_bn_momentum must be in (0,1].")


@dataclass(frozen=True)
class LSTMPolicyConfig:
    """Backbone scheduling and memory feedback during LSTM detector evaluation."""
    mode: str = "fixed"
    interval: int = 8
    scene_change_threshold: float | None = None
    reset_on_scene_change: bool = True
    memory_feedback: str = "clean"

    def __post_init__(self):
        """Validate inference mode, big-call interval, scene threshold, and memory feedback."""
        if self.mode not in ("small", "big", "fixed") or self.interval < 1:
            raise ValueError("Invalid mode or interval.")
        if self.scene_change_threshold is not None:
            probability(self.scene_change_threshold, "Scene-change threshold")
        if self.memory_feedback not in ("clean", "none"):
            raise ValueError("memory_feedback must be clean or none.")


@dataclass(frozen=True)
class LSTMEvalConfig:
    """Device, output filtering, timing warmup, and reporting controls for LSTM evaluation."""
    device: str = "cpu"
    score_threshold: float = 0.001
    max_detections: int = 100
    warmup_frames: int = 3
    seed: int = 0
    progress: bool = True
    compute_losses: bool = True

    def __post_init__(self):
        """Validate score filtering, detection limits, and timing warmup counts."""
        probability(self.score_threshold, "Evaluation score threshold")
        if self.max_detections < 1 or self.warmup_frames < 0:
            raise ValueError("Invalid detection limit or warmup count.")
