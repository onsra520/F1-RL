"""Print actual WP7 readiness evidence; exit 2 while training is blocked."""
import json
import argparse
from pathlib import Path

from f1rl.research.wp7_admission import evaluate_admission
from f1rl.research.wp7_run_manifest import read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-software", action="store_true", help="run focused registered software checks without training")
    parser.add_argument('--verify-execution', action='store_true', help='execute the fixed TRAIN acceptance cohort without PPO')
    parser.add_argument('--execution-output', type=Path, help='new external directory for execution certificates')
    parser.add_argument('--run-manifest', type=Path, help='prepared or artifact-bound run manifest')
    parser.add_argument('--review', type=Path, help='run review envelope; repository trust registry is authoritative')
    parser.add_argument('--profile',choices=('pv5_v1','spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'),default='pv5_v1')
    args = parser.parse_args()
    if args.review and not args.run_manifest:
        parser.error('--review requires --run-manifest')
    if args.verify_execution != (args.execution_output is not None):
        parser.error('--verify-execution and --execution-output must be supplied together')
    report = evaluate_admission(Path(__file__).resolve().parents[2],
        verify_software=args.verify_software, profile=args.profile,
        verify_execution=args.verify_execution, execution_output=args.execution_output,
        run_manifest=read_json(args.run_manifest) if args.run_manifest else None,
        review_envelope=read_json(args.review) if args.review else None)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["training_authorized"] else 2)
