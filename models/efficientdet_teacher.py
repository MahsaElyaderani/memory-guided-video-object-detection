"""Compatibility import; implementation lives in models/frozen_efficientdet_teacher.py."""
import sys
from . import frozen_efficientdet_teacher as _implementation

sys.modules[__name__] = _implementation
