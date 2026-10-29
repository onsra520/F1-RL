"""Legacy CLI for TRAIN-only scripted/PPO driving diagnostics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.driving_diagnostics import (
    diagnostic_action, trace_episode, benchmark, main,
)


if __name__ == "__main__":
    main()
