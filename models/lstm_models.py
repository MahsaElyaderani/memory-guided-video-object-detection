"""Model definitions and construction for standalone and joint LSTM experiments."""
import torch
from torch import nn

from lstm_config import PairedConfig
from models.classification_pretraining import JointClassifier
from models.joint_lstm_detector import JointLSTMDetector
from models.lstm_ssd_detector import LSTMSSD
from models.ssd_head import SSDFeatureMaps, SSDBoxPredictor
from utils.joint_initialization import make_joint_config

EXPERIMENTS = {'small_050': .5, 'big_120d': 1.2, 'joint_120d': 1.2,
               'small_tinynet_d': 1., 'big_efficientnet_b0': 1.,
               'joint_efficientnet_b0_tinynet_d': 1.}


class MatchedDetector(JointLSTMDetector):
    def __init__(self, experiment, seed=0):
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown or archived experiment: {experiment}')
        family = ('tinynet_d' if experiment == 'small_tinynet_d' else
                  'efficientnet_b0' if 'efficientnet_b0' in experiment else 'timm_mobilenet_v2')
        cfg = make_joint_config(family, EXPERIMENTS[experiment])
        paired = PairedConfig(fast_width_mult=.5 if family == 'timm_mobilenet_v2' else 1.)
        self.standalone = None if experiment.startswith('joint_') else ('small' if experiment.startswith('small_') else 'big')
        if self.standalone is None:
            super().__init__(LSTMSSD(cfg), paired)
        else:
            nn.Module.__init__(self)
            self.big = LSTMSSD(cfg)
            self.paired_config = paired
            self.fast_backbone = nn.Identity()
            self.small_projection = nn.Identity()
            if self.standalone == 'small':
                self.big.backbone.set_output_stride16()
        # Separate initialization stream: changing backbone depth cannot change shared weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.big.lstm.reset_parameters()
            self.big.feature_maps = SSDFeatureMaps()
            self.big.predictor = SSDBoxPredictor(cfg.feature_channels, cfg.num_classes,
                self.ssd_cfg.num_anchors_per_location, self.ssd_cfg.include_background_class)
        self.configure_detector_training()

    def forward_big(self, frames, state=None):
        if self.standalone == 'small':
            return self.big._forward_from_backbone(self.small_projection(self.big.backbone(frames.small)), state, True,
                                                   anchor_dtype=frames.dtype)
        return super().forward_big(frames, state)

    def forward_small(self, frames, state=None):
        return self.forward_big(frames, state) if self.standalone else super().forward_small(frames, state)

    def forward_selected(self, frames, state, use_big, small_commit=None):
        if self.standalone:
            return self.forward_big(frames, state)
        return super().forward_selected(frames, state, use_big, small_commit)


def build_model(experiment, task, seed):
    detector = MatchedDetector(experiment, seed)
    if task == 'detection':
        return detector
    if task != 'classification':
        raise ValueError('Unknown matched task.')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 1)
        return JointClassifier(detector.model_cfg, detector.paired_config, 30, detector=detector)
