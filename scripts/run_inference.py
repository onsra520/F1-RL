"""Run the saved PPO policy on TRAIN Straight, without creating a training session."""
from pathlib import Path
from importlib import import_module
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces.runtime_selection import ensure_runtime
from src.interfaces.inference_cli import main as _main

def main(argv=None):
    return _main(ROOT, argv, torch_factory=lambda: import_module('torch'))

if __name__ == '__main__':
    raise SystemExit(main())
