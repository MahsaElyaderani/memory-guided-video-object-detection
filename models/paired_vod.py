"""Compatibility import; implementation lives in models/joint_lstm_detector.py."""
import sys
from . import joint_lstm_detector as _implementation

sys.modules[__name__] = _implementation
