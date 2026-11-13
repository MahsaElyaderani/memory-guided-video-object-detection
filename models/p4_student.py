"""TinyNet-D through blocks.4, with optional fixed-addition keyframe memory."""
import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig, SSDConfig
from models.anchors import SSDAnchorGenerator
from models.backbones import TimmMobileNetV2Backbone
from models.ssd_head import SSDBoxPredictor, SSDFeatureMaps


class P4Student(nn.Module):
    def __init__(self, memory, *, imagenet=False, detector=None):
        super().__init__()
        if memory not in ('none', 'p4'):
            raise ValueError('Expected none or p4 memory.')
        self.memory = memory
        backbone = TimmMobileNetV2Backbone(model_name='tinynet_d.in1k')
        if imagenet:
            backbone.load_pretrained()
        if detector is not None:
            source = detector.big.backbone if detector.standalone else detector.fast_backbone
            if source.model_name != backbone.model_name:
                raise ValueError('Detector does not contain the expected TinyNet-D weights.')
            backbone.network.load_state_dict(source.network.state_dict(), strict=True)
        net = backbone.network
        self.stem = nn.Sequential(net.conv_stem, net.bn1)
        self.blocks = net.blocks[:5]
        self.register_buffer('mean', backbone.input_mean)
        self.register_buffer('std', backbone.input_std)
        self.projection = nn.Conv2d(64, 256, 1)
        self.feature_maps = SSDFeatureMaps()
        self.ssd_cfg = SSDConfig()
        self.predictor = SSDBoxPredictor(ModelConfig().feature_channels, 30,
                                         self.ssd_cfg.num_anchors_per_location, True)
        self.anchors = SSDAnchorGenerator()
        if detector is not None:
            if detector.ssd_cfg != self.ssd_cfg:
                raise ValueError('Incompatible pretrained SSD configuration.')
            self.feature_maps.load_state_dict(detector.big.feature_maps.state_dict())
            self.predictor.load_state_dict(detector.big.predictor.state_dict())

    def forward(self, frames, cache=None):
        if frames.shape[1:] != (3, 160, 160):
            raise ValueError('Expected 160 x 160 student RGB views.')
        student = self.blocks(self.stem(((frames + 1) * .5 - self.mean) / self.std))
        if self.memory == 'p4':
            keys, values, valid = cache
            query = F.normalize(student.flatten(2).transpose(1, 2), dim=-1)
            # Cosine logits with fixed inverse temperature sqrt(64).
            scores = (query @ keys) * 8.
            weights = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
            student = student + (weights @ values).transpose(1, 2).reshape_as(student)
        features = self.feature_maps(self.projection(student))
        logits, boxes = self.predictor(features)
        anchors = self.anchors.generate([f.shape[-2:] for f in features], device=frames.device, dtype=frames.dtype)
        return dict(class_logits=logits, box_regression=boxes, anchors=anchors)
