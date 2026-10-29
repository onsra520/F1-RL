"""Legacy CLI for TRAIN-development policy comparison. No scientific admission."""
from pathlib import Path
from importlib import import_module
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.driving_comparison import SEEDS, metrics, improved, main as _main


def main():
    return _main(torch_factory=lambda: import_module('torch'))


if __name__ == "__main__":
    raise SystemExit(main())
