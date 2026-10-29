"""Audit stored WP7 real PPO runs, without changing original model evidence.

Use --run for calibration and confirmation. Writes a NEW audit file outside
source worktree and does not access HELD-OUT tracks.
"""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evidence import audit_trajectories as _implementation
from scripts.evidence.audit_trajectories import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/evidence/audit_trajectories.py")
    raise SystemExit(main())
