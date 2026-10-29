"""Generate/check SHA256 evidence manifest for *independently restored* files.

No Google Drive credentials or network operations are included. Creating a
manifest is not proof of an off-device backup or independent restore.

Examples (PowerShell):
  python scripts/wp7_evidence_manifest.py create --source "D:\\archive\\run" --manifest "D:\\archive\\run-manifest.json"
  python scripts/wp7_evidence_manifest.py verify --source "D:\\restore\\run" --manifest "D:\\archive\\run-manifest.json"
"""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.evidence import evidence_manifest as _implementation
from scripts.evidence.evidence_manifest import main


def __getattr__(name):
    """Forward historical helper imports to the grouped implementation."""
    return getattr(_implementation, name)


if __name__ == "__main__":
    # Display the stable CLI name even when invoked through the old filename.
    sys.argv[0] = str(_REPOSITORY_ROOT / "scripts/evidence/evidence_manifest.py")
    raise SystemExit(main())
