"""Compatibility import; configuration definitions live in lstm_config.py."""
import sys
import lstm_config as _implementation

sys.modules[__name__] = _implementation
