import hashlib
import json
from pathlib import Path

import pytest

from f1rl.local_demo.runner import run_m5_cycle2


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/research/m5-cycle2-c0-slalom-v1.json"


def _plan(**updates):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    value = {
        "schema_version": "f1rl_m5_cycle2_training_plan_v1",
        "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        "parent_checkpoint_sha256": profile["parent_checkpoint_sha256"],
        "training_authorized_by_owner_request_sha256": profile["owner_request_sha256"],
        "track_ids": profile["tracks"],
        "max_additional_timesteps": profile["max_additional_timesteps"],
        "checkpoint_interval": profile["checkpoint_interval"],
        "device": "cpu", "heldout_access": False, "scientific_evidence": False,
    }
    value.update(updates)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    value["sha256"] = hashlib.sha256(payload).hexdigest()
    return value


def _invoke(tmp_path: Path, plan: dict):
    path = tmp_path / "cycle2-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    output = tmp_path / "run-output"
    return run_m5_cycle2(ROOT, plan_path=path, output_directory=output,
                         checkpoint_observer=lambda *_: True)


def test_cycle2_profile_is_exactly_five_train_tracks_and_excludes_right_turn_v1():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["tracks"] == [
        "straight-sprint-v1", "slalom-v1", "right-turn-train-v2", "ramp-v1",
        "left-turn-train-v2",
    ]
    assert "right-turn-v1" not in profile["tracks"]
    assert profile["device"] == "cpu"
    assert profile["max_additional_timesteps"] == 32768


def test_cycle2_admission_rejects_right_turn_v1_before_issuing_capability(tmp_path):
    plan = _plan(track_ids=["straight-sprint-v1", "slalom-v1", "right-turn-v1",
                            "ramp-v1", "left-turn-train-v2"])
    with pytest.raises(ValueError, match="source-bound plan/profile"):
        _invoke(tmp_path, plan)
    assert not (tmp_path / "run-output").exists()


def test_cycle2_admission_rejects_budget_above_owner_cap(tmp_path):
    plan = _plan(max_additional_timesteps=32769)
    with pytest.raises(ValueError, match="source-bound plan/profile"):
        _invoke(tmp_path, plan)
    assert not (tmp_path / "run-output").exists()


def test_cycle2_admission_rejects_non_cpu_profile(tmp_path):
    plan = _plan(device="cuda")
    with pytest.raises(ValueError, match="source-bound plan/profile"):
        _invoke(tmp_path, plan)
    assert not (tmp_path / "run-output").exists()


def test_cycle2_rejects_output_resolving_inside_repository(tmp_path):
    output = ROOT.parent / 'elsewhere' / '..' / ROOT.name / 'cycle2-output'
    with pytest.raises(ValueError, match='external evidence directory'):
        run_m5_cycle2(ROOT, plan_path=tmp_path / 'missing.json', output_directory=output,
                      checkpoint_observer=lambda *_: True)
