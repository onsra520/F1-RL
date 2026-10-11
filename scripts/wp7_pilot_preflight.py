"""Read-only WP7 pilot preflight. Exit 2 means scientific review gates are blocked."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1rl.learning.pilot_preflight import build_pilot_preflight


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=root/"configs/recovery/wp7-c2-candidate.yml")
    parser.add_argument("--profile", type=Path, default=root/"configs/recovery/wp7-pilot-200k.yml")
    args = parser.parse_args()
    report = build_pilot_preflight(
        repository_root=root,
        candidate_path=args.candidate,
        runtime_config_path=args.profile,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2 if report["review_blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
