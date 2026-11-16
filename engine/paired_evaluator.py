"""Compatibility import; implementation lives in engine/lstm_evaluation.py."""
import sys
from . import lstm_evaluation as _implementation

sys.modules[__name__] = _implementation
