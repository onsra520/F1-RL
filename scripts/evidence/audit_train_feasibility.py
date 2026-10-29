"""Print deterministic body-aware TRAIN evidence; no policy/model is loaded."""
import json
import argparse
from pathlib import Path



from src.validation.wp7_track_feasibility import audit_train_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-alternatives", action="store_true",
                        help="certify bounded planar candidates separately from authored-path findings")
    parser.add_argument('--registered-center-gates',action='store_true',
                        help='explicit owner 5A center-gate search; preserves full-body road/obstacle checks')
    parser.add_argument('--execute-controls',action='store_true',help='execute source-bound nominal TRAIN control tapes')
    parser.add_argument('--profile',choices=('pv5_v1','spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3'),default='pv5_v1')
    args = parser.parse_args()
    print(json.dumps(audit_train_tracks(Path(__file__).resolve().parents[2],
                    search_alternatives=args.search_alternatives or args.registered_center_gates,
                    execute_controls=args.execute_controls,registered_center_gates=args.registered_center_gates,execution_profile=args.profile), indent=2, allow_nan=False))
