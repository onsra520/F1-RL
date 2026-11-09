"""TRAIN-5-v2 source binding and fail-closed admission; no PPO evaluation."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.research import wp7_train5_v2_protocol as p

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=Path(r"D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106")
REG=ROOT/p.REGISTRATION


def _verify():
    return p.validate_preview(ROOT,registration_file=REG,evidence_root=EVIDENCE)


def test_default_train4_identity_and_separate_train5_v2():
    assert TRAIN_IDS==("straight-sprint-v1","slalom-v1","right-turn-v1","ramp-v1")
    assert p.TRAIN5_IDS==(
        "straight-sprint-v1","slalom-v1","right-turn-train-v2",
        "ramp-v1","left-turn-train-v2")
    assert p.HELDOUT_IDS==("left-turn-v1","offset-slalom-v1")
    assert p.TRAIN5_IDS!=TRAIN_IDS
    assert p.HISTORICAL_IDS==TRAIN_IDS


def test_source_and_external_real_v4_witness_preflight_valid():
    reg=_verify()
    assert reg["status"]=="draft_not_authorized"
    assert reg["preflight_digest"]==p._digest(
        {k:v for k,v in reg.items() if k!="preflight_digest"})
    assert len(reg["source_sha256"])==44
    assert len(reg["evidence_sha256"])==4
    assert reg["runtime_profile"]==p.RUNTIME_SPEC
    for rel,digest in reg["source_sha256"].items():
        historical=json.loads((ROOT/p.HISTORICAL_REGISTRATION).read_text("utf8"))
        assert digest==historical["source_sha256"][rel]
        blob=subprocess.run(["git","-C",str(ROOT),"show",
            f"{reg['source_anchor_commit']}:{rel}"],check=True,capture_output=True).stdout
        assert sha256(blob).hexdigest()==reg["git_blob_sha256"][rel]


def test_preview_builds_five_source_bound_v4_factories_no_heldout_access(monkeypatch):
    import f1rl.learning.reconstruction as reconstruction
    original=reconstruction.load_track_asset
    opened=[]
    def record(path):
        if "/evaluation/" in Path(path).as_posix():
            pytest.fail("HELD-OUT geometry file was opened")
        opened.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(reconstruction,"load_track_asset",record)
    rows=p.preview_train5_v2(ROOT,registration_file=REG,evidence_root=EVIDENCE)
    assert tuple(x.track_id for x in rows.scenarios)==p.TRAIN5_IDS
    assert len(opened)==9  # 5 v2 candidate + 4 legacy template TRAIN
    assert all(name in opened for name in (
        "right-turn-train-v2.json","left-turn-train-v2.json",
        "right-turn-v1.json"))
    for scenario in rows.scenarios:
        fac=scenario.factory
        assert fac.reset_config is None
        assert fac.reset_track_id is None
        assert fac.physics_config.ground_motion_version==p.RUNTIME_SPEC["physics_version"]
        assert fac.race_config.version==p.RUNTIME_SPEC["race_version"]
        assert fac.physics_config.independent_suspension_policy.max_abs_tilt_rad==.7
        assert fac.requires_scientific_admission
        assert fac.action_contract.version=="steering-throttle-brake-v2"


def test_dev_seed_proposals_are_disjoint_and_never_misrepresented_as_unused_globally():
    doc=_verify()
    proposal=doc["proposed_training_seeds"]
    assert tuple(proposal)==tuple(p.PROPOSED)
    values=[v for cohort in proposal.values() for row in cohort for v in row.values()]
    assert len(values)==18 and len(set(values))==18
    assert all(isinstance(v,int) and 0<=v<2**32 for v in values)
    assert doc["seed_audit_status"]=="PROPOSED_UNVERIFIED_GLOBAL"
    assert doc["global_seed_registry_receipt"] is None
    assert doc["owner_approval"] is None
    assert doc["proposed_design"]["proposed_development_cells"]==60
    assert doc["proposed_design"]["proposed_development_episodes"]==3600
    assert doc["training_permission"] is False


@pytest.mark.parametrize("gate",[
 p.require_training_admission,
 p.require_smoke_admission,
 p.require_heldout_admission,
])
def test_running_unapproved_train_smoke_or_heldout_is_unconditionally_denied(gate):
    with pytest.raises(PermissionError,match="CLOSED"):
        gate(ROOT,REG,owner_approval="forged",skip_seed_audit=True)


@pytest.mark.parametrize("field,value",[
    ("status","approved"),
    ("training_permission",True),
    ("smoke_permission",True),
    ("heldout_permission",True),
    ("seed_audit_status","GLOBAL_VERIFIED"),
    ("owner_approval","fake signed"),
    ("global_seed_registry_receipt","fake"),
    ("train_ids",["left-turn-v1","right-turn-train-v2"]),
    ("heldout_ids",[]),
    ("runtime_profile",{"physics_version":"spatial_rotating_compliant_lifecycle_v3"}),
    ("baseline_head","not-history"),
    ("candidate","configs/recovery/wp7-c2-candidate.yml"),
])
def test_in_memory_proposal_tampering_fails_before_geometry(monkeypatch,field,value):
    original=Path.read_text
    tampered=json.loads(REG.read_text("utf-8"))
    tampered[field]=value
    # Even with updated canonical digest, semantic constraints must reject it.
    tampered["preflight_digest"]=p._digest(
       {k:v for k,v in tampered.items() if k!="preflight_digest"})
    def f(path,*args,**kwargs):
        if Path(path).resolve()==REG.resolve():
            return json.dumps(tampered)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",f)
    import f1rl.learning.reconstruction as reconstruction
    monkeypatch.setattr(reconstruction,"load_track_asset",
                        lambda *_:pytest.fail("preflight accessed geometry"))
    with pytest.raises(p.Train5PreflightError):
        _verify()


def test_in_memory_locked_file_hash_mismatch_is_caught_before_geometry(monkeypatch):
    original=Path.read_text
    tampered=json.loads(REG.read_text("utf-8"))
    tampered["git_blob_sha256"]["assets/tracks/training/right-turn-train-v2.json"]="0"*64
    tampered["preflight_digest"]=p._digest(
       {k:v for k,v in tampered.items() if k!="preflight_digest"})
    def f(path,*args,**kwargs):
        if Path(path).resolve()==REG.resolve():
            return json.dumps(tampered)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",f)
    import f1rl.learning.reconstruction as reconstruction
    monkeypatch.setattr(reconstruction,"load_track_asset",
                        lambda *_:pytest.fail("tampered source lock loaded geometry"))
    with pytest.raises(p.Train5PreflightError,match="canonical Git blob hash mismatch"):
        _verify()


def test_registration_location_must_be_pinned(tmp_path):
    with pytest.raises((p.Train5PreflightError,FileNotFoundError)):
        p.validate_preview(ROOT,registration_file=REG.parent/"wrong.json",evidence_root=EVIDENCE)


def test_missing_external_step3_witness_fails_closed(tmp_path):
    with pytest.raises((p.Train5PreflightError,FileNotFoundError)):
        p.validate_preview(ROOT,registration_file=REG,evidence_root=tmp_path)


def test_v1_dangerous_identity_not_present_in_new_roster():
    orig=json.loads((ROOT/"assets/tracks/manifest.json").read_text("utf-8"))
    pair=json.loads((ROOT/p.MANIFEST).read_text("utf-8"))
    assert pair["tracks"][:2]==orig["tracks"][:2]
    assert pair["tracks"][3]==orig["tracks"][3]
    assert pair["tracks"][5:]==orig["tracks"][4:]
    assert "left-turn-train-v1" not in str(pair)
    assert p._manifest_contract(ROOT)==pair["tracks"]
