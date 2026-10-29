"""Compatibility checks for WP3 artifact encoding and restart behavior."""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

from src.interfaces import _artifacts, campaign
from src.interfaces import evaluation
from src.interfaces.evaluation import build_evaluation_roster
from src.validation.simulation_only_protocol import load_simulation_only_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_json_and_roster_hashes_are_stable():
    assert _artifacts._canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        _artifacts._canonical({"not_finite": float("nan")})

    config, _, _ = campaign._load_campaign_config(ROOT)
    simulation, _ = load_simulation_only_manifest(ROOT)
    schedule = campaign.model_schedule(config, simulation)
    manifest = {
        "sha256": "a" * 64,
        "source_commit": "b" * 40,
        "simulation_only_config_sha256": "c" * 64,
        "registration_sha256": "d" * 64,
        "track_manifest_sha256": "e" * 64,
        "execution_order": schedule,
    }
    roster = build_evaluation_roster(ROOT, manifest, config)
    assert roster["sha256"] == "a5413e53bc61da5365888034082a8372c2dd903d9b7cc3c66f7a3c4b86c008f9"
    assert sha256(_artifacts._canonical(roster)).hexdigest() == (
        "8ad6160268c489be472752eb9ac8a49a4e9acd9fff04906d6ffc5518fb9ccb2f"
    )


def test_strict_json_preserves_bytes_and_rejects_duplicate_or_non_finite_values(tmp_path):
    path = tmp_path / "artifact.json"
    original = b'{\n  "stable": true\n}\n'
    path.write_bytes(original)
    value, raw = _artifacts._strict_json(path)
    assert value == {"stable": True}
    assert raw == original

    path.write_text('{"key":1,"key":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key: key"):
        _artifacts._strict_json(path)
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        _artifacts._strict_json(path)
    path.write_text('[1,2]', encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        _artifacts._strict_json(path)


def test_json_writes_keep_exact_format_exclusive_create_and_atomic_replace(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    _artifacts._write_new_json(path, {"z": 2, "a": 1})
    expected = b'{\n  "a": 1,\n  "z": 2\n}\n'
    assert path.read_bytes() == expected
    with pytest.raises(FileExistsError):
        _artifacts._write_new_json(path, {"a": 9})
    assert path.read_bytes() == expected

    _artifacts._write_immutable_json(path, {"z": 2, "a": 1})
    with pytest.raises(ValueError, match="refusing to overwrite immutable WP3 artifact"):
        _artifacts._write_immutable_json(path, {"a": 9})
    assert path.read_bytes() == expected

    original_replace = os.replace

    def fail_replace(source, target):
        raise OSError("atomic replace failed")

    monkeypatch.setattr(_artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="atomic replace failed"):
        _artifacts._atomic_json(path, {"new": True})
    assert path.read_bytes() == expected
    monkeypatch.setattr(_artifacts.os, "replace", original_replace)
    _artifacts._atomic_json(path, {"new": True})
    assert path.read_bytes() == b'{\n  "new": true\n}\n'


def test_completed_resume_verifies_final_inventory_without_starting_children(tmp_path, monkeypatch):
    output = tmp_path / "campaign-output"
    output.mkdir()
    (output / "WP3-EVIDENCE-MANIFEST.json").write_text("{}\n", encoding="utf-8")
    state = {"status": "COMPLETE", "completed_model_ids": ["all"]}
    monkeypatch.setattr(campaign, "_load_campaign",
                        lambda *_: ({}, {}, [], state, output))
    verified = []
    monkeypatch.setattr(campaign, "_verify_evidence_manifest",
                        lambda path: verified.append(path))
    monkeypatch.setattr(campaign, "_run_child",
                        lambda *args, **kwargs: pytest.fail("completed resume launched a child"))

    result = campaign.run_campaign(ROOT, output, resume=True)

    assert result is state
    assert verified == [output]
    assert (output / ".campaign.lock").read_bytes() == b"0"


def test_resume_refuses_running_model_without_validated_run_evidence(tmp_path, monkeypatch):
    output = tmp_path / "campaign-output"
    output.mkdir()
    state = {
        "status": "RUNNING",
        "current_model_id": "C-seed0",
        "completed_model_ids": [],
        "models": {"C-seed0": {"status": "running"}},
    }
    schedule = [{"model_id": "C-seed0", "run_relative_path": "runs/C-seed0"}]
    monkeypatch.setattr(campaign, "_load_campaign",
                        lambda *_: ({}, {"training": {"timesteps_per_model": 1}},
                                    schedule, state, output))
    monkeypatch.setattr(campaign, "_run_child",
                        lambda *args, **kwargs: pytest.fail("resume retried an incomplete model"))

    with pytest.raises(RuntimeError, match="not retryable; refusing duplicate training"):
        campaign.run_campaign(ROOT, output, resume=True)
    assert state["models"]["C-seed0"]["status"] == "running"


def test_interrupted_evaluation_cell_is_invalidated_once_and_never_redrawn(tmp_path):
    event_path = tmp_path / "cell.jsonl"
    model = {"model_id": "C-seed0", "candidate_id": "C", "seed_index": 0}
    evaluation._append_event(event_path, {"event": "cell_started", "model_id": "C-seed0"})
    evaluation._append_event(event_path, {"event": "episode_started", "episode": 7})

    recovered = evaluation._validate_prior_cell(
        event_path, model, "straight-sprint-v1", "nominal")
    persisted = event_path.read_bytes()
    replayed = evaluation._validate_prior_cell(
        event_path, model, "straight-sprint-v1", "nominal")

    assert recovered == replayed
    assert recovered["status"] == "invalid_aborted"
    assert recovered["invalid_episodes"] == 1
    assert recovered["invalid_episode"]["error_type"] == "InterruptedEvaluation"
    assert len(persisted.splitlines()) == 4
    assert event_path.read_bytes() == persisted
