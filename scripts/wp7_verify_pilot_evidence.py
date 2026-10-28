"""Read-only post-copy checksum verification of WP7 MOCK pilot evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys as _entry_sys
from pathlib import Path as _EntryPath

_entry_root = str(_EntryPath(__file__).resolve().parents[1])
if _entry_root not in _entry_sys.path:
    _entry_sys.path.insert(0, _entry_root)

from src.rl.pilot_milestones import verify_pilot_evidence_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing evaluations/manifest.json")
    args = parser.parse_args()
    result = verify_pilot_evidence_manifest(args.run_directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
