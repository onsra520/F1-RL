"""Read-only prospective WP7 owner-decision preregistration preflight.

Examples:
python scripts/wp7_review_owner_acceptance.py
python scripts/wp7_review_owner_acceptance.py --require-ready  # deliberately exits 2
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.learning.owner_acceptance import review_preregistered_contract

DEFAULT_CONTRACT = Path("configs/recovery/wp7-owner-decisions-preregistered-v1.yml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--require-ready", action="store_true")
    options = parser.parse_args()
    review = review_preregistered_contract(options.contract)
    print(json.dumps(asdict(review), indent=2))
    return 2 if options.require_ready and not review.training_permitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
