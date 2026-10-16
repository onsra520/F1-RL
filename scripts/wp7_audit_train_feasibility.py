"""Print deterministic body-aware TRAIN evidence; no policy/model is loaded."""
import json
import argparse
from pathlib import Path

from f1rl.research.wp7_track_feasibility import audit_train_tracks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-alternatives", action="store_true",
                        help="certify bounded planar candidates separately from authored-path findings")
    args = parser.parse_args()
    print(json.dumps(audit_train_tracks(Path(__file__).resolve().parents[1],
                    search_alternatives=args.search_alternatives), indent=2, allow_nan=False))
