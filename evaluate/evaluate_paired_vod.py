"""Compatibility entry point; implementation lives in root evaluate_lstm_models.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "evaluate_lstm_models.py"), run_name="__main__")
else:
    import evaluate_lstm_models as _implementation
    sys.modules[__name__] = _implementation
