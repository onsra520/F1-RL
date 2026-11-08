"""Read-only truthful TRAIN-5 local engineering audit. No permission mutation."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


class AuditError(ValueError):
    pass


def load_local_inputs(root: Path) -> dict:
    root = Path(root)
    def load(path: str):
        return json.loads((root / path).read_text(encoding="utf8"))
    return {
        "matrix": load("configs/research/wp7-core-readiness-matrix-v1.json"),
        "registry": load("configs/research/wp7-train5-global-seed-registry-owner-exception-20261107.json"),
        "stage2": load("configs/research/wp7-train5-stage2-science-proposal-v1.json"),
        "v4": load("configs/research/wp7-train5-v4-local-checkpoint-draft.json"),
    }


def build_local_audit(root: Path, p: dict) -> dict:
    root = Path(root)
    matrix, reg, stage2, v4 = (p[k] for k in ("matrix", "registry", "stage2", "v4"))
    subs = matrix["subsystems"]
    if len(subs) != 9 or len({x["id"] for x in subs}) != 9:
        raise AuditError("unexpected compatibility subsystem registry")
    capabilities = [c for s in subs for c in s["capabilities"]]
    counts = Counter(c["status"] for c in capabilities)
    open_gates = sum(s["compatibility"]["status"] == "in_progress" for s in subs)
    if any(s["compatibility"]["status"] != "in_progress" for s in subs):
        raise AuditError("unverified compatibility status promotion")
    missing = sorted({
        path for c in capabilities for path in (c["source"], c["test"])
        if not (root / path).is_file()
    })
    seed_list = reg["candidate_seeds"]
    if len(seed_list) != 18 or len(set(seed_list)) != len(seed_list):
        raise AuditError("duplicate or missing reserved training seeds")
    exception = reg["exception"]
    if (exception["affected_run_count"] != 15 or
        exception["job_logs_not_recovered"] is not True or
        exception["all_15_runs_are_accepted_as_PASS"] is not False or
        reg["universal_global_seed_nonuse_independently_proven"] is not False):
        raise AuditError("historical 15-job limitation misrepresented")
    if (reg["registry_gate_status"] != "CLOSED_WITH_OWNER_ACCEPTED_EVIDENCE_EXCEPTION"
        or stage2["training_permission"] is not False
        or stage2["heldout_permission"] is not False
        or stage2["owner_protocol_signed"] is not False
        or v4["permissions"]["ppo_training"] is not False
        or v4["permissions"]["heldout"] is not False):
        raise AuditError("invalid science admission or owner state")
    files = (
        "configs/research/wp7-train5-v4-local-checkpoint-draft.json",
        "f1rl/research/wp7_train5_v4_budget.py",
        "f1rl/research/wp7_train5_v4_dev_metrics.py",
    )
    return {
        "schema": "f1rl_wp7_train5_local_unsigned_gate_audit_v1",
        "scope": "LOCAL_SOURCE_ONLY_NO_SCIENTIFIC_EXECUTION",
        "matrix": {
            "capability_total": len(capabilities),
            "implemented": counts["implemented"],
            "blocked": counts["blocked"],
            "compatibility_open": open_gates,
            "all_source_and_test_paths_exist": not missing,
            "missing_paths": missing,
            "subsystems": [s["id"] for s in subs],
            "formal_gate_verdict": "NOT_CLOSED_NO_C2_TRAJECTORY_CROSS_OS_RECEIPTS",
        },
        "seeds": {
            "registered_count": len(seed_list),
            "distinct_count": len(set(seed_list)),
            "missing_gha_job_logs": 15,
            "accepted_exception": True,
            "absolute_nonuse_proven": False,
            "reserved_consumed_by_this_audit": 0,
        },
        "v4_local_source_sha256_lf": {
            f: sha256((root / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in files
        },
        "admission": {
            "ready": False,
            "ppo_training_permitted": False,
            "heldout_permitted": False,
            "owner_signature": False,
            "independent_scientific_review": False,
            "reason": "9 compatibility gates; 3 science capabilities; signature and authentic scientific approval absent",
        },
    }
