"""Compatibility entry point; use train_attention_models.py for new commands."""
import sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("train_attention_models", run_name="__main__")
else:
    import train_attention_models as _implementation
    sys.modules[__name__] = _implementation
