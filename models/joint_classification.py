"""Compatibility import; implementation lives in models/classification_pretraining.py."""
import sys
from . import classification_pretraining as _implementation

sys.modules[__name__] = _implementation
