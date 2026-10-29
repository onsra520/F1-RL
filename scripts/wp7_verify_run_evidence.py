"""Verify independently retained run bytes and optional trusted signatures.

No host runtime recapture, model loading, training or authorization occurs.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.run_evidence_workflow import read_json, verify_run_manifest, verify_run_reviews, main

if __name__ == "__main__":
    raise SystemExit(main())
