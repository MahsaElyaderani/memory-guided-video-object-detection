"""Shared data container: two RGB resolutions of the same frame, before backbone normalization."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PairedFrames:
    """Carry aligned 320x320 and 160x160 RGB tensors with matching batch/time axes."""
    big: torch.Tensor
    small: torch.Tensor

    def __post_init__(self):
        """Validate view sizes and shared leading dimensions, device, and dtype."""
        if (self.big.shape[-3:] != (3, 320, 320) or self.small.shape[-3:] != (3, 160, 160)
                or self.big.shape[:-3] != self.small.shape[:-3]):
            raise ValueError('Paired RGB views require matching batch/time axes and sizes 320/160.')
        if self.big.device != self.small.device or self.big.dtype != self.small.dtype:
            raise ValueError('Paired views must have the same device and dtype.')

    # Training indexes batch/time axes; spatial axes refer to the big view.
    @property
    def shape(self):
        """Report the big view shape; its leading batch/time axes match the small view."""
        return self.big.shape

    @property
    def ndim(self):
        """Return the number of tensor dimensions shared by both views."""
        return self.big.ndim

    @property
    def device(self):
        """Return the device shared by both views."""
        return self.big.device

    @property
    def dtype(self):
        """Return the dtype shared by both views."""
        return self.big.dtype

    def __len__(self):
        """Return the leading dimension, such as batch size or clip length."""
        return len(self.big)

    def __getitem__(self, index):
        """Apply the same batch/time indexing to both views."""
        return PairedFrames(self.big[index], self.small[index])

    def to(self, *args, **kwargs):
        """Move or convert both tensors together using Tensor.to arguments."""
        return PairedFrames(self.big.to(*args, **kwargs), self.small.to(*args, **kwargs))

    def detach(self):
        """Return both views detached from their computation graphs."""
        return PairedFrames(self.big.detach(), self.small.detach())

    def cpu(self):
        """Move both views to CPU."""
        return self.to('cpu')

    def pin_memory(self):
        """Pin both CPU tensors for DataLoader transfers."""
        return PairedFrames(self.big.pin_memory(), self.small.pin_memory())


def big_frames(frames):
    """Extract the big view, or pass through an ordinary image tensor."""
    return frames.big if isinstance(frames, PairedFrames) else frames


def stack_frames(frames, dim=0):
    """Stack tensors or stack each paired view separately along the requested axis."""
    if isinstance(frames[0], PairedFrames):
        if not all(isinstance(frame, PairedFrames) for frame in frames):
            raise ValueError('Cannot mix single and paired views in a batch.')
        return PairedFrames(torch.stack([f.big for f in frames], dim),
                            torch.stack([f.small for f in frames], dim))
    return torch.stack(list(frames), dim)
