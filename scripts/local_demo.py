"""Explicit Windows-only simulation research demo bootstrap; no scientific permit."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.training import local_simulation as _implementation
from scripts.training.local_simulation import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/training/local_simulation.py")
    raise SystemExit(main())
