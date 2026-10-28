"""Read-only post-copy checksum verification of WP7 MOCK pilot evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning.pilot_milestones import verify_pilot_evidence_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing evaluations/manifest.json")
    args = parser.parse_args()
    result = verify_pilot_evidence_manifest(args.run_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
