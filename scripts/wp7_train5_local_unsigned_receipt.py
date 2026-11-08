"""Produce a transparent unsigned local TRAIN-5 status receipt (no training)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from f1rl.research.wp7_train5_v4_local_audit import build_local_audit, load_local_inputs
from f1rl.research.wp7_train5_v4_budget import load_v4_budget, verify_pinned_pilot_file

root = Path(__file__).resolve().parents[1]
target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "wp7-local-unsigned-receipt.json"
protocol = load_v4_budget(root)
verify_pinned_pilot_file(root, protocol)
data = build_local_audit(root, load_local_inputs(root))
data["observed_utc"] = datetime.now(timezone.utc).isoformat()
data["head_commit"] = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
data["uncommitted"] = bool(subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=root, text=True
).strip())
data["source_commit_or_ci_pass_claim"] = False
data["signed_owner_admission_claim"] = False
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", "utf8")
print(f"LOCAL AUDIT ONLY; ADMISSION={data['admission']['ready']}")
print(f"RECEIPT={target}")
