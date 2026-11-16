"""Compatibility import; implementation lives in models/lstm_ssd_detector.py."""
import sys
from . import lstm_ssd_detector as _implementation

sys.modules[__name__] = _implementation
