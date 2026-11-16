"""Compatibility import; use lstm_checkpoints or runtime for new code."""
import sys
from . import lstm_checkpoints as _implementation

sys.modules[__name__] = _implementation
