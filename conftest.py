"""Ensure the repo root is importable so `parking` and `train` resolve in tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
