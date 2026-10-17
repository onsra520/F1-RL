"""Print a nominal TRAIN control-tape witness or independently replay a pinned one."""
import argparse
import json
from pathlib import Path

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.research.wp7_executed_feasibility import (
    certify_planar_execution, certify_ramp_execution, certify_coupled_ramp_execution, replay_planar_execution,
)
from f1rl.research.wp7_protocol import _unique


def _nonfinite(value):
    raise ValueError(f'nonfinite JSON value: {value}')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--track',choices=TRAIN_IDS)
    mode.add_argument('--replay',type=Path)
    mode.add_argument('--ramp-proposal',action='store_true',help='explicit unadmitted road-relative engineering profile')
    mode.add_argument('--ramp-coupled-proposal',action='store_true',help='explicit unadmitted coupled force/work profile')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.replay is not None:
        if args.expected_sha256 is None:
            parser.error('--replay requires an independently retained --expected-sha256')
        document = json.loads(args.replay.read_text(encoding='utf-8'),object_pairs_hook=_unique,parse_constant=_nonfinite)
        result = replay_planar_execution(root,document,expected_sha256=args.expected_sha256)
    else:
        if args.expected_sha256 is not None:
            parser.error('--expected-sha256 requires --replay')
        result = (certify_coupled_ramp_execution(root) if args.ramp_coupled_proposal else
                  certify_ramp_execution(root) if args.ramp_proposal else certify_planar_execution(root,args.track))
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status'] in ('executable_numerical_planar_path','executable_numerical_ramp_path') else 2


if __name__=='__main__':
    raise SystemExit(main())
