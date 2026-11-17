"""Compatibility import; implementation lives in routing/backbone_schedule.py."""
import sys
from . import backbone_schedule as _implementation

sys.modules[__name__] = _implementation
