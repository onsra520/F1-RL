"""Compatibility CLI for frozen-model/FULL-06 integrity verification."""
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.validation.frozen_asset_integrity import digest, require, verify_frozen_assets

# Preserve historical module constants for Python callers of this executable.
ROOT = REPOSITORY_ROOT / "docs/research/evidence"
MODEL = ROOT / "wp3-issue51-frozen-final-models-20261028"
FULL = ROOT / "wp3-issue51-full06-traindev-20261027"


def main() -> None:
    result = verify_frozen_assets(REPOSITORY_ROOT)
    print(f"PASS: {result['models']} models, {result['reports']} reports and {result['cell_logs']} cell logs")


if __name__ == "__main__":
    main()
