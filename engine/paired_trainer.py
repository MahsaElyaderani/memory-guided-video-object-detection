"""Compatibility import; implementation lives in engine/lstm_training.py."""
import sys
from . import lstm_training as _implementation

sys.modules[__name__] = _implementation
