"""Print a nominal TRAIN control-tape witness or independently replay a pinned one."""
import argparse
import json
from pathlib import Path

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.research.wp7_executed_feasibility import (
    certify_planar_execution, certify_ramp_execution, certify_coupled_ramp_execution, replay_planar_execution,
    certify_registered_center_execution,
    certify_spatial_ramp_execution, certify_spatial_profile_execution,
)
from f1rl.research.wp7_protocol import _unique


def _nonfinite(value):
    raise ValueError(f'nonfinite JSON value: {value}')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--track',choices=TRAIN_IDS)
    mode.add_argument('--registered-center-track',choices=TRAIN_IDS,help='registered physics and owner directed-center gates')
    mode.add_argument('--spatial-profile-track',choices=TRAIN_IDS,help='explicit draft spatial TRAIN registration')
    mode.add_argument('--replay',type=Path)
    mode.add_argument('--ramp-proposal',action='store_true',help='explicit unadmitted road-relative engineering profile')
    mode.add_argument('--ramp-coupled-proposal',action='store_true',help='explicit unadmitted coupled force/work profile')
    mode.add_argument('--ramp-spatial-proposal',action='store_true',help='explicit unadmitted spatial force/contact profile')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--profile',choices=('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'))
    parser.add_argument('--seed-index',type=int)
    parser.add_argument('--reset-rank',type=int)
    parser.add_argument('--reset-index',type=int)
    args = parser.parse_args()
    if args.spatial_profile_track is None and any(v is not None for v in (args.seed_index,args.reset_rank,args.reset_index,args.profile)):
        parser.error('seed/reset selectors require --spatial-profile-track')
    root = Path(__file__).resolve().parents[1]
    if args.replay is not None:
        if args.expected_sha256 is None:
            parser.error('--replay requires an independently retained --expected-sha256')
        document = json.loads(args.replay.read_text(encoding='utf-8'),object_pairs_hook=_unique,parse_constant=_nonfinite)
        result = replay_planar_execution(root,document,expected_sha256=args.expected_sha256)
    else:
        if args.expected_sha256 is not None:
            parser.error('--expected-sha256 requires --replay')
        result = (certify_spatial_profile_execution(root,args.spatial_profile_track,
                    seed_index=0 if args.seed_index is None else args.seed_index,
                    reset_rank=0 if args.reset_rank is None else args.reset_rank,reset_index=args.reset_index,
                    profile=args.profile or 'spatial_ground_v1') if args.spatial_profile_track else
                  certify_registered_center_execution(root,args.registered_center_track) if args.registered_center_track else
                  certify_spatial_ramp_execution(root) if args.ramp_spatial_proposal else
                  certify_coupled_ramp_execution(root) if args.ramp_coupled_proposal else
                  certify_ramp_execution(root) if args.ramp_proposal else certify_planar_execution(root,args.track))
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['status'] in ('executable_numerical_planar_path','executable_numerical_ramp_path',
                                   'executable_numerical_registered_center_path','executable_numerical_registered_spatial_path') else 2


if __name__=='__main__':
    raise SystemExit(main())
