"""Model definitions and construction for standalone and joint LSTM experiments."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor

from utils.config import LSTMConfig, ModelConfig, SSDConfig
from datasets.frame_views import PairedFrames, big_frames
from models.backbones import build_backbone, Backbone
from models.bottleneck_lstm import BottleneckLSTMCell, State
from models.anchors import SSDAnchorGenerator
from models.ssd_head import SSDFeaturePyramid, SSDPredictionHeads

EXPERIMENTS = {'small_050': .5, 'big_120d': 1.2, 'joint_120d': 1.2,
               'small_tinynet_d': 1., 'big_efficientnet_b0': 1.,
               'joint_efficientnet_b0_tinynet_d': 1.}


def select_memory(previous, new, commit):
    """Commit candidate (c,h) states only for samples whose boolean commit mask is true."""
    if len(previous) != 2 or len(new) != 2:
        raise ValueError("A recurrent state is the pair (c,h).")
    if previous[0].shape != new[0].shape or previous[1].shape != new[1].shape:
        raise ValueError("Previous and new state shapes differ.")
    mask = torch.as_tensor(commit, device=new[0].device)
    if mask.dtype != torch.bool or mask.shape != (new[0].shape[0],):
        raise ValueError("Memory commit must be a boolean vector [B].")
    return tuple(torch.where(mask[:, None, None, None], b, a) for a, b in zip(previous, new))


class LSTMDetectorCore(nn.Module):
    """Single-Bottleneck-LSTM detector.

    One frame:
      RGB -> timm backbone -> Bottleneck-LSTM -> 5 SSD feature levels
          -> class logits and box offsets.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        ssd_cfg: SSDConfig | None = None,
    ) -> None:
        """Build the backbone, recurrent cell, SSD pyramid, prediction heads, and anchors."""
        super().__init__()
        self.model_cfg = model_cfg or ModelConfig()
        self.ssd_cfg = ssd_cfg or SSDConfig()

        self.backbone = build_backbone(self.model_cfg)
        self.lstm = BottleneckLSTMCell(
            input_channels=self.model_cfg.mobilenet_output_channels,
            hidden_channels=self.model_cfg.lstm_channels,
            kernel_size=self.model_cfg.lstm_kernel_size,
            forget_bias=self.model_cfg.lstm_forget_bias,
            activation=self.model_cfg.lstm_activation,
        )
        self.feature_maps = SSDFeaturePyramid()
        self.predictor = SSDPredictionHeads(
            feature_channels=self.model_cfg.feature_channels,
            num_classes=self.model_cfg.num_classes,
            anchors_per_location=self.ssd_cfg.num_anchors_per_location,
            include_background_class=self.ssd_cfg.include_background_class,
        )
        self.anchor_generator = SSDAnchorGenerator(
            num_levels=self.ssd_cfg.num_levels,
            min_scale=self.ssd_cfg.min_scale,
            max_scale=self.ssd_cfg.max_scale,
            aspect_ratios=self.ssd_cfg.aspect_ratios,
            interpolated_scale_aspect_ratio=self.ssd_cfg.interpolated_scale_aspect_ratio,
            reduce_boxes_in_lowest_layer=self.ssd_cfg.reduce_boxes_in_lowest_layer,
        )

    def _validate_frame(self, frame: Tensor) -> None:
        """Check RGB batch shape and configured input resolution."""
        if frame.ndim != 4 or frame.shape[1] != 3:
            raise ValueError(f"Expected frame [B,3,H,W], got {tuple(frame.shape)}")
        if frame.shape[-2:] != (self.model_cfg.image_size, self.model_cfg.image_size):
            raise ValueError(
                f"Expected {self.model_cfg.image_size}x{self.model_cfg.image_size} input; "
                f"got {tuple(frame.shape[-2:])}"
            )

    def forward_step(
        self,
        frame: Tensor,
        state: State | None = None,
    ) -> Tuple[Dict[str, Tensor | List[Tensor]], State]:
        """Process one frame while carrying recurrent state online."""
        self._validate_frame(frame)
        backbone_feature = self.backbone(frame)
        return self._forward_from_backbone(
            backbone_feature, state, anchor_dtype=frame.dtype
        )

    def _forward_from_backbone(
        self,
        backbone_feature: Tensor,
        state: State | None,
        *,
        anchor_dtype: torch.dtype,
    ) -> Tuple[Dict[str, Tensor | List[Tensor]], State]:
        """Run aligned backbone features through memory and SSD, returning predictions and new state."""
        if backbone_feature.shape[1:] != (self.model_cfg.mobilenet_output_channels, self.model_cfg.image_size // 32, self.model_cfg.image_size // 32):
            raise ValueError("timm ConvLSTM feature shape does not match the configured input size.")
        lstm_feature, new_state = self.lstm(backbone_feature, state)
        features = self.feature_maps(lstm_feature)
        class_logits, box_regression = self.predictor(features)

        output: Dict[str, Tensor | List[Tensor]] = {
            "class_logits": class_logits,
            "box_regression": box_regression,
            "features": features,
        }

        shapes = [(f.shape[-2], f.shape[-1]) for f in features]
        anchors = self.anchor_generator.generate(
            shapes, device=backbone_feature.device, dtype=anchor_dtype
        )
        if anchors.shape[0] != class_logits.shape[1]:
            raise RuntimeError(
                "Anchor/prediction count mismatch: "
                f"anchors={anchors.shape[0]}, predictions={class_logits.shape[1]}"
            )
        output["anchors"] = anchors

        return output, new_state


class LSTMDetector(nn.Module):
    """Standalone or joint detector with LSTM memory and SSD heads."""
    def __init__(self, big: LSTMDetectorCore, paired_config=None, *, standalone=None):
        """Register the primary core and optional small branch without duplicating shared heads."""
        super().__init__()
        if not isinstance(big, LSTMDetectorCore):
            raise TypeError("The big branch must be an LSTMDetectorCore instance.")
        self.standalone = standalone
        self.big = big
        self.paired_config = paired_config or LSTMConfig(
            fast_width_mult=(1.0 if big.model_cfg.backbone == 'efficientnet_b0' else
                             .5),
            fast_image_size=big.model_cfg.image_size // 2)
        if standalone is not None:
            if standalone not in ('small', 'big'):
                raise ValueError('Standalone must be small, big, or None for joint.')
            self.fast_backbone = nn.Identity()
            self.small_projection = nn.Identity()
            if standalone == 'small':
                self.big.backbone.set_output_stride16()
            return
        cfg = self.paired_config
        if big.model_cfg.image_size != 320 or cfg.fast_image_size != 160:
            raise ValueError("Joint inputs must be big 320 and small 160.")
        if big.model_cfg.num_classes != 30 or not big.ssd_cfg.include_background_class:
            raise ValueError("This joint implementation requires the native 30-class VID detector.")
        if big.model_cfg.backbone == 'efficientnet_b0':
            if cfg.fast_width_mult != 1.0 or cfg.min_depth != 8:
                raise ValueError('EfficientNet-B0 pairs only with unscaled TinyNet-D.')
            self.fast_backbone = Backbone(1.0, 16, model_name='tinynet_d.in1k')
        elif big.model_cfg.backbone == 'tinynet_d':
            raise ValueError('This backbone is supported as a matched standalone small model.')
        elif big.model_cfg.uses_timm:
            if cfg.fast_width_mult not in (.5, 1.) or cfg.min_depth != 8:
                raise ValueError("The timm small branch requires width .5/1.0 and min_depth 8.")
            self.fast_backbone = Backbone(cfg.fast_width_mult, 16)
        else:
            raise ValueError(f'Unsupported joint backbone: {big.model_cfg.backbone}')
        self.small_projection = nn.Identity()
        self.configure_detector_training()

    @property
    def model_cfg(self):
        """Expose the primary core architecture settings."""
        return self.big.model_cfg

    @property
    def ssd_cfg(self):
        """Expose the shared SSD and anchor settings."""
        return self.big.ssd_cfg

    def train(self, mode=True):
        """Apply model mode and the configured backbone BatchNorm behavior."""
        super().train(mode)
        if mode and self.paired_config.freeze_bn_stats:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if mode and self.paired_config.fast_bn_momentum is not None:
            for module in self.fast_backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.momentum = self.paired_config.fast_bn_momentum
                    module.train(True)
        return self

    def configure_detector_training(self):
        """Enable all detector gradients and apply its training-mode policy."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self.train(True)

    def initial_state(self, frames):
        """Allocate zero shared memory matching the configured feature grid."""
        self.big._validate_frame(big_frames(frames))
        side = self.model_cfg.image_size // 32
        return self.big.lstm.init_state(len(frames), side, side, frames.device, frames.dtype)

    def fast_features(self, frames):
        """Extract small-branch features from independently resized 160px RGB views."""
        self.big._validate_frame(big_frames(frames))
        if not isinstance(frames, PairedFrames):
            raise ValueError("timm small inference requires PairedFrames resized independently from the source RGB frame.")
        return self.small_projection(self.fast_backbone(frames.small))

    def forward_big(self, frames, state=None):
        """Run the primary detector; standalone-small models use the small input view."""
        if self.standalone == 'small':
            return self.big._forward_from_backbone(self.small_projection(self.big.backbone(frames.small)), state,
                                                   anchor_dtype=frames.dtype)
        return self.big.forward_step(big_frames(frames), state)

    def forward_small(self, frames, state=None):
        """Run the small branch through shared memory/SSD, or the sole standalone detector."""
        if self.standalone:
            return self.forward_big(frames, state)
        feature = self.fast_features(frames)
        return self.big._forward_from_backbone(feature, state, anchor_dtype=frames.dtype)

    def forward_selected(self, frames, state, use_big, small_commit=None):
        """Select backbone features per sample and commit only the permitted memory updates."""
        if self.standalone:
            return self.forward_big(frames, state)
        mask = torch.as_tensor(use_big, device=frames.device)
        if mask.dtype != torch.bool or mask.shape != (len(frames),):
            raise ValueError("use_big must be a boolean vector [B].")
        state = self.initial_state(frames) if state is None else state
        large, small = self.big.backbone(big_frames(frames)), self.fast_features(frames)
        if large.shape != small.shape:
            raise RuntimeError("Aligned big and small features must have identical shapes.")
        features = torch.where(mask[:, None, None, None], large, small)
        output, new_state = self.big._forward_from_backbone(features, state,
                                                           anchor_dtype=frames.dtype)
        commit = mask
        if small_commit is not None:
            own = torch.as_tensor(small_commit, device=frames.device)
            if own.dtype != torch.bool or own.shape != mask.shape:
                raise ValueError("small_commit must be a boolean vector [B].")
            commit = mask | own
        return output, select_memory(state, new_state, commit)


def build_model(experiment, task, seed):
    # Classification reuses these detector definitions; import it only when constructing a model.
    """Build one experiment with seeded shared initialization and the requested task head."""
    from models.classifier import PretrainClassifier

    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown or archived experiment: {experiment}')
    family = ('tinynet_d' if experiment == 'small_tinynet_d' else
              'efficientnet_b0' if 'efficientnet_b0' in experiment else 'timm_mobilenet_v2')
    cfg = ModelConfig(backbone=family, width_mult=EXPERIMENTS[experiment], image_size=320)
    paired = LSTMConfig(fast_width_mult=.5 if family == 'timm_mobilenet_v2' else 1.)
    standalone = None if experiment.startswith('joint_') else ('small' if experiment.startswith('small_') else 'big')
    detector = LSTMDetector(LSTMDetectorCore(cfg), paired, standalone=standalone)
    # Preserve the separate initialization stream for shared LSTM/SSD weights.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        detector.big.lstm.reset_parameters()
        detector.big.feature_maps = SSDFeaturePyramid()
        detector.big.predictor = SSDPredictionHeads(cfg.feature_channels, cfg.num_classes,
            detector.ssd_cfg.num_anchors_per_location, detector.ssd_cfg.include_background_class)
    detector.configure_detector_training()
    if task == 'detection':
        return detector
    if task != 'classification':
        raise ValueError('Unknown matched task.')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 1)
        return PretrainClassifier(detector.model_cfg, detector.paired_config, 30, detector=detector)
