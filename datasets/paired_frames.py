"""Compatibility import; implementation lives in datasets/frame_views.py."""
import sys
from . import frame_views as _implementation

sys.modules[__name__] = _implementation
