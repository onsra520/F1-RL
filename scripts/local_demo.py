"""Explicit Windows-only simulation research demo bootstrap; no scientific permit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces.local_demo_cli import main as _main

def main():
    return _main(ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
