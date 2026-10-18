"""Pytest setup: make app/core importable for every test file."""

import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "app" / "core"

if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))