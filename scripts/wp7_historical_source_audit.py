"""Read-only source provenance report for *historical* WP7 PPO archives.

Usage:
    python scripts/wp7_historical_source_audit.py --archive-run RUN --output OUT.json

Never rewrites a training archive, opens HELD-OUT, runs PPO or asserts
Scientific Acceptance. An exact Git commit source match cannot recover
unknown training-time Python packages or GPU settings.
"""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evidence import historical_source_audit as _implementation
from scripts.evidence.historical_source_audit import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/evidence/historical_source_audit.py")
    raise SystemExit(main())
