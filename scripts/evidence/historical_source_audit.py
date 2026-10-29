"""Read-only source provenance report for *historical* WP7 PPO archives.

Usage:
    python scripts/wp7_historical_source_audit.py --archive-run RUN --output OUT.json

Never rewrites a training archive, opens HELD-OUT, runs PPO or asserts
Scientific Acceptance. An exact Git commit source match cannot recover
unknown training-time Python packages or GPU settings.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.historical_source_audit_workflow import audit_historical_manifest_file, main as _main

def main() -> int:
    return _main(ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
