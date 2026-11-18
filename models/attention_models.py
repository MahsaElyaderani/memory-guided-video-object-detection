"""Model definition: TinyNet-D through blocks.4 and SSD, with optional cached-keyframe attention memory."""
import torch
from torch import nn
from torch.nn import functional as F

from utils.config import ModelConfig, SSDConfig
from models.anchors import SSDAnchorGenerator
from models.backbones import Backbone
from models.ssd_head import SSDPredictionHeads, SSDFeaturePyramid


class AttentionDetector(nn.Module):
    """TinyNet-D/SSD student with optional spatial attention to cached teacher P4."""
    def __init__(self, memory, *, imagenet=False):
        """Build the 64-channel student and fresh SSD heads; optionally load ImageNet weights."""
        super().__init__()
        if memory not in ('none', 'p4'):
            raise ValueError('Expected none or p4 memory.')
        self.memory = memory
        backbone = Backbone(model_name='tinynet_d.in1k')
        if imagenet:
            backbone.load_pretrained()
        net = backbone.network
        self.stem = nn.Sequential(net.conv_stem, net.bn1)
        # blocks.4 already supplies 64 channels at stride 16 for the memory comparison.
        self.blocks = net.blocks[:5]
        self.register_buffer('mean', backbone.input_mean)
        self.register_buffer('std', backbone.input_std)
        self.feature_maps = SSDFeaturePyramid(input_channels=64)
        self.ssd_cfg = SSDConfig()
        self.predictor = SSDPredictionHeads((64, *ModelConfig().feature_channels[1:]), 30,
                                         self.ssd_cfg.num_anchors_per_location, True)
        self.anchors = SSDAnchorGenerator()

    def forward(self, frames, cache=None):
        """Return SSD logits, offsets, and anchors from 160px views and optional teacher cache."""
        if frames.shape[1:] != (3, 160, 160):
            raise ValueError('Expected 160 x 160 student RGB views.')
        student = self.blocks(self.stem(((frames + 1) * .5 - self.mean) / self.std))
        if self.memory == 'p4':
            keys, values, valid = cache
            # Each of 100 student locations reads the valid P4 locations. Normalize for
            # matching, then add the retrieved raw values on the current frame grid.
            query = F.normalize(student.flatten(2).transpose(1, 2), dim=-1)
            # Cosine logits with fixed inverse temperature sqrt(64).
            scores = (query @ keys) * 8.
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            student = student + (weights @ values).transpose(1, 2).reshape_as(student)
        features = self.feature_maps(student)
        logits, boxes = self.predictor(features)
        anchors = self.anchors.generate([f.shape[-2:] for f in features], device=frames.device, dtype=frames.dtype)
        return dict(class_logits=logits, box_regression=boxes, anchors=anchors)
