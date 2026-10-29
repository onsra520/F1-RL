"""Legacy CLI for TRAIN-only scripted/PPO driving diagnostics."""
from pathlib import Path
from importlib import import_module
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.driving_diagnostics import (
    diagnostic_action, trace_episode, benchmark as _benchmark, main as _main,
)


def benchmark(factory, model, threads, output):
    return _benchmark(factory, model, threads, output, torch_runtime=import_module('torch'))


def main():
    return _main(torch_factory=lambda: import_module('torch'))


if __name__ == "__main__":
    main()
