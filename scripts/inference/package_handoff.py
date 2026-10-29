"""Package a proven exploratory final model; never starts training."""
from pathlib import Path
from importlib import import_module
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.interfaces.demo_handoff import save, main as _main

def main(argv=None):
    return _main(ROOT, argv, torch_factory=lambda: import_module('torch'))

if __name__ == '__main__':
    raise SystemExit(main())
