"""Print actual WP7 readiness evidence; exit 2 while training is blocked."""
import json
import argparse
from pathlib import Path

from f1rl.research.wp7_admission import evaluate_admission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-software", action="store_true", help="run bounded synthetic checks without training")
    parser.add_argument('--profile',choices=('pv5_v1','spatial_ground_v1','spatial_owner_v2'),default='pv5_v1')
    args = parser.parse_args()
    report = evaluate_admission(Path(__file__).resolve().parents[1], verify_software=args.verify_software,profile=args.profile)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["training_authorized"] else 2)
