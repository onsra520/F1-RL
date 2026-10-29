"""Generate/check SHA256 evidence manifest for *independently restored* files.

No Google Drive credentials or network operations are included. Creating a
manifest is not proof of an off-device backup or independent restore.

Examples (PowerShell):
  python scripts/wp7_evidence_manifest.py create --source "D:\\archive\\run" --manifest "D:\\archive\\run-manifest.json"
  python scripts/wp7_evidence_manifest.py verify --source "D:\\restore\\run" --manifest "D:\\archive\\run-manifest.json"
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.evidence_manifest_workflow import _unique_pairs, main
from src.rl.evidence_backup import build_evidence_manifest, serialize_manifest, verify_evidence_manifest

if __name__ == "__main__":
    raise SystemExit(main())
