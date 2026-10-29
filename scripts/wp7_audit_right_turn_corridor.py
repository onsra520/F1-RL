"""Replay only the archived Right Turn TRAIN final PPO models for road-union audit."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evidence import audit_road_corridor as _implementation
from scripts.evidence.audit_road_corridor import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/evidence/audit_road_corridor.py")
    raise SystemExit(main())
