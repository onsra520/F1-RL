"""Run the saved PPO policy on TRAIN Straight, without creating a training session."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.inference import run_policy as _implementation
from scripts.inference.run_policy import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/inference/run_policy.py")
    raise SystemExit(main())
