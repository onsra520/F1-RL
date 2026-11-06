"""CI: source-bound TRAIN-5 engineering preparation, never evaluation/training.

GitHub runner cannot access signed external Step3 evidence. Check all on-repo
source hashes, exact new TRAIN metadata, proposed seeds, and immutable gates.
Do not label this proof as full scientific admission.
"""
from pathlib import Path
from hashlib import sha256
import json

import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from f1rl.research import wp7_train5_v2_protocol as protocol
REG=ROOT/protocol.REGISTRATION
UNAVAILABLE=ROOT/"_external_witness_not_checked_in"
doc=protocol.validate_preview(ROOT,registration_file=REG,
                              evidence_root=UNAVAILABLE,check_evidence=False)
assert not doc["training_permission"]
assert not doc["smoke_permission"]
assert not doc["heldout_permission"]
assert doc["seed_audit_status"]=="PROPOSED_UNVERIFIED_GLOBAL"
assert doc["global_seed_registry_receipt"] is None
assert doc["owner_approval"] is None
for gate in (protocol.require_training_admission,
             protocol.require_smoke_admission,
             protocol.require_heldout_admission):
    try:
        gate()
    except PermissionError:
        pass
    else:
        raise AssertionError("A draft protocol exposed an execution route")
print("CI_DRAFT_PREFLIGHT_OK",len(doc["source_sha256"]),"source hashes")
print("EXTERNAL_STEP3_EVIDENCE","NOT_VERIFIED_ON_GITHUB_RUNNER")
print("TRAINING_SMOKE_HELDOUT","ALL_BLOCKED")
