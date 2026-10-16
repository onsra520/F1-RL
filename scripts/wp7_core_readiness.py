"""Read-only engineering readiness matrix; never scientific admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1rl.research.wp7_readiness import evaluate_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="run software evidence tests (no research PPO)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = evaluate_readiness(root, verify=args.verify)
    print(json.dumps(result, indent=2))
    return 0 if result["all_subsystems_functional_90"] and result["compatibility_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
