"""Pytest setup: make app/core, app/ (main.py) and frontend/ importable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for folder in (ROOT / "frontend", ROOT / "app", ROOT / "app" / "core"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))