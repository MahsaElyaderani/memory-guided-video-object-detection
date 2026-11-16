"""Compatibility import; implementation lives in models/attention_memory_detector.py."""
import sys
from . import attention_memory_detector as _implementation

sys.modules[__name__] = _implementation
