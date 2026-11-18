"""Frozen teacher adapter: loads EfficientDet and extracts keyframe detections and P4 memory features."""
import runpy
import sys
import numpy as np
import torch
from torch.nn import functional as F


class EfficientDetTeacher:
    """Frozen external EfficientDet adapter providing keyframe detections and P4 memory."""
    def __init__(self, root, checkpoint, device):
        # The fork's script imports its sibling tracking module by bare name.
        """Load the Delta fork checkpoint on the selected device and freeze all teacher weights."""
        script = root.resolve() / 'baseline/evaluate_tracking.py'
        paths = sys.path.copy()
        try:
            sys.path.insert(0, str(script.parent))
            baseline = runpy.run_path(str(script))
        finally:
            sys.path[:] = paths
        self.device = device
        self.net = baseline['load_model'](checkpoint, device).requires_grad_(False).eval()
        baseline['set_delta_distillation_schedule'](self.net, 1)
        self.transform = baseline['transforms_coco_eval'](self.net.config.image_size, use_prefetcher=True)
        self.records = baseline['records']

    @torch.no_grad()
    def keyframe(self, images, *, detect=False):
        """Return P4 keys [B,64,1024], values [B,1024,64], validity mask, and optional detections."""
        sizes = [im.size for im in images]
        batch = [self.transform(im, {'img_size': im.size}) for im in images]
        pixels = torch.from_numpy(np.stack([p for p, _ in batch])).to(self.device, torch.float32)
        mean = pixels.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        pixels = (pixels - mean * 255) / (std * 255)
        scales = pixels.new_tensor([target['img_scale'] for _, target in batch])
        body = self.net.model
        detections = None
        if detect:
            # FPN outputs start at P3. Capture P4 from the same pass that produces keyframe boxes.
            captured = []
            hook = body.fpn.register_forward_hook(lambda mod, inputs, outputs: captured.append(outputs[1]))
            try:
                detections = self.net(pixels, scales, pixels.new_tensor(sizes)).cpu().numpy()
            finally:
                hook.remove()
            p4 = captured[0]
        else:
            p4 = body.fpn(body.backbone(pixels))[1]
        if p4.shape[1:] != (64, 32, 32):
            raise ValueError('Expected 512-pixel EfficientDet-D0 teacher with 64-channel P4.')
        # Top-left letterbox. Keep cells overlapping resized content, including partial boundary cells.
        step = torch.arange(32, device=pixels.device) * 16
        extent = [(int(w * min(512/w, 512/h)), int(h * min(512/w, 512/h))) for w, h in sizes]
        widths, heights = pixels.new_tensor(extent).unbind(1)
        valid = (step[None, :, None] < heights[:, None, None]) & (step[None, None, :] < widths[:, None, None])
        values = p4.flatten(2).transpose(1, 2)
        return (F.normalize(values, dim=-1).transpose(1, 2), values, valid.flatten(1)), detections
