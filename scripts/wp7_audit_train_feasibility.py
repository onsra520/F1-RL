"""Print deterministic body-aware TRAIN evidence; no policy/model is loaded."""
import json
from pathlib import Path

from f1rl.research.wp7_track_feasibility import audit_train_tracks


if __name__ == "__main__":
    print(json.dumps(audit_train_tracks(Path(__file__).resolve().parents[1]), indent=2, allow_nan=False))
