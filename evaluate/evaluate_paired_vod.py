"""Compatibility import; implementation lives in evaluate/evaluate_lstm_models.py."""
import sys
from pathlib import Path

if __name__ == "__main__":
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_path(str(Path(__file__).with_name("evaluate_lstm_models.py")), run_name="__main__")
else:
    from . import evaluate_lstm_models as _implementation
    sys.modules[__name__] = _implementation
