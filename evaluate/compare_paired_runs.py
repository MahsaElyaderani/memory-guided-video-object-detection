"""Compatibility entry point; implementation lives in compare_detection_runs.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __package__ in (None, ""):
    import compare_detection_runs as _implementation
else:
    from . import compare_detection_runs as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
