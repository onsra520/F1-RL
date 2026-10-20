from pathlib import Path
import json

import pytest

from f1rl.research.simulation_validation import (
    load_simulation_validation_protocol,
    run_simulation_validation,
)

ROOT = Path(__file__).resolve().parents[2]


def test_versioned_simulation_validation_runs_repeatable_real_train_transitions():
    protocol, digest = load_simulation_validation_protocol(ROOT)
    result = run_simulation_validation(ROOT)

    assert protocol["schema_version"] == "f1rl_simulation_validation_v1"
    assert len(protocol["coverage_matrix"]) == 6
    assert len(digest) == 64
    assert result["status"] == "SIMULATION_CONSISTENCY_PASS"
    assert result["physical_calibration_claim"] is False
    assert result["scientific_acceptance_claim"] is False
    assert result["registered_4m_boundary_preserved"] is True
    assert set(result["track_results"]) == set(protocol["registered_tracks"])
    assert all(row["status"] == "PASS" and row["steps"] == 4
               for row in result["track_results"].values())


def test_simulation_validation_contract_rejects_missing_coverage_reference(tmp_path):
    destination = tmp_path / "configs" / "research" / "simulation-validation-v1.json"
    destination.parent.mkdir(parents=True)
    protocol = json.loads((ROOT / "configs" / "research" / "simulation-validation-v1.json").read_text(
        encoding="utf-8"))
    protocol["coverage_matrix"][3]["tests"] = ["tests/does-not-exist.py"]
    destination.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage matrix"):
        load_simulation_validation_protocol(tmp_path)


@pytest.mark.parametrize("boundary", ["reset", "transition"])
def test_simulation_replay_rejects_changed_reset_protocol_identity(monkeypatch, boundary):
    from f1rl.learning.gym_environment import F1RLEnv

    if boundary == "reset":
        original = F1RLEnv.reset

        def spoofed(self, *args, **kwargs):
            observation, info = original(self, *args, **kwargs)
            info = dict(info)
            info["train_reset"] = {**info["train_reset"], "version": "spoofed"}
            return observation, info

        monkeypatch.setattr(F1RLEnv, "reset", spoofed)
    else:
        original = F1RLEnv.step

        def spoofed(self, action):
            observation, reward, terminated, truncated, info = original(self, action)
            info = dict(info)
            info["train_reset"] = {**info["train_reset"], "version": "spoofed"}
            return observation, reward, terminated, truncated, info

        monkeypatch.setattr(F1RLEnv, "step", spoofed)

    with pytest.raises(ValueError, match="reset_identity_mismatch"):
        run_simulation_validation(ROOT)
