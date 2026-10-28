"""Local-only entry point for the frozen WP3 training/evaluation campaign."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_demo.campaign import main


if __name__ == "__main__":
    raise SystemExit(main())
