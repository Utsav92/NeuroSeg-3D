"""Lets scripts run from a fresh checkout without `pip install -e .` (adds repo root to sys.path)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
