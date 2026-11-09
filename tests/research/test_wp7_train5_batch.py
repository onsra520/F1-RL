from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
import pytest

from scripts.wp7_train5_batch import build_plan, sign_plan

ROOT = Path(__file__).resolve().parents[2]


def test_unsigned_plan_covers_exact_six_runs_without_creating_output(tmp_path):
    output = tmp_path / "runs"
    plan = build_plan(ROOT, output, Path(r"D:\unused-evidence"), valid_hours=168)
    assert len(plan["runs"]) == 6
    assert [item["initial_checkpoint"] for item in plan["runs"]] == [None, None, None, "C-seed0", "B-seed1", "A-seed2"]
    assert plan["budget"]["actual_final_steps"] == 200704
    assert plan["output_root"] == str(output.resolve())
    assert plan["authorized"] is False
    assert not output.exists()


def test_fake_owner_signs_six_distinct_scoped_requests(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from scripts import wp7_train5_batch as batch

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    envelopes = sign_plan(ROOT, plan, key, now=datetime(2026, 11, 9, tzinfo=timezone.utc))
    assert len(envelopes) == 6
    assert len({item["payload"]["nonce"] for item in envelopes}) == 6
    assert [(e["payload"]["cohort"], e["payload"]["replicate"]) for e in envelopes] == [
        (run["cohort"], run["replicate"]) for run in plan["runs"]]
    assert all(e["payload"]["manifest_sha256"] == plan["manifest_sha256"] for e in envelopes)
    assert not (tmp_path / "runs").exists()


def test_fake_terminal_signing_creates_packet_without_starting_training(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from scripts import wp7_train5_batch as batch
    import sys

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    plan_path, key_path, packet_path = tmp_path / "plan.json", tmp_path / "fake-key", tmp_path / "packet.json"
    plan_path.write_text(json.dumps(plan), encoding="utf8")
    key_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()))
    monkeypatch.setattr("builtins.input", lambda prompt: "DONG Y TRAIN5 NOI BO")
    monkeypatch.setattr(batch.getpass, "getpass", lambda prompt: "")
    monkeypatch.setattr(sys, "argv", ["wp7_train5_batch", "sign", "--plan", str(plan_path),
                                      "--private-key", str(key_path), "--packet", str(packet_path)])
    batch.main()
    packet = json.loads(packet_path.read_text(encoding="utf8"))
    assert len(packet["envelopes"]) == 6
    assert not (tmp_path / "runs").exists()


def test_signing_with_wrong_owner_key_is_rejected(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from scripts import wp7_train5_batch as batch

    enrolled = Ed25519PrivateKey.generate()
    wrong = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: enrolled.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    with pytest.raises(ValueError, match="signing key"):
        sign_plan(ROOT, plan, wrong)
    assert not (tmp_path / "runs").exists()


def test_batch_rejects_one_bad_signature_before_any_run(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from scripts import wp7_train5_batch as batch

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    requests = sign_plan(ROOT, plan, key)
    requests[-1]["signature_hex"] = "00" * 64
    with pytest.raises(ValueError, match="signature"):
        batch.execute_batch(ROOT, plan, requests)
    assert not (tmp_path / "runs").exists()


def test_fake_batch_journals_six_runs_without_changing_frozen_models(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from f1rl.research import wp7_train5_v4_checkpoint as checkpoints
    from scripts import wp7_train5_batch as batch
    import scripts.wp7_train5_v4_factory_preview as preview

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    monkeypatch.setattr(preview, "preview_registered_factories", lambda *args, **kwargs: object())
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    envelopes = sign_plan(ROOT, plan, key)

    class FakeRunner:
        def __init__(self, output, cohort, replicate):
            self.output, self.cohort, self.replicate = output, cohort, replicate

        def run_authorized_multi(self, scenarios, *, envelope, evidence_root):
            execution.verify_run(ROOT, self.output, self.cohort, self.replicate, envelope)
            directory = self.output / f"{self.cohort}-{self.replicate}"
            directory.mkdir()
            model = directory / "fake-final.zip"
            model.write_bytes(f"{self.cohort}-{self.replicate}".encode())
            return SimpleNamespace(final_model_path=model, run_directory=directory, actual_total_timesteps=200704)

    monkeypatch.setattr(checkpoints, "prepare_v4_registered_runner", lambda root, output, **run: FakeRunner(output, **run))
    result = batch.execute_batch(ROOT, plan, envelopes)
    events = [json.loads(line) for line in (tmp_path / "runs" / "batch-journal.jsonl").read_text().splitlines()]
    assert len(result["runs"]) == 6
    assert [event["event"] for event in events] == ["started", "completed"] * 6
    assert result["frozen_checkpoint_sha256"] == checkpoints.FROZEN_CHECKPOINTS
    assert result["scientific_admission"] is False


def test_batch_stops_after_failed_run_without_starting_later_runs(tmp_path, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution
    from f1rl.research import wp7_train5_v4_checkpoint as checkpoints
    from scripts import wp7_train5_batch as batch
    import scripts.wp7_train5_v4_factory_preview as preview

    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    monkeypatch.setattr(batch, "_require_clean_source", lambda root, head: None)
    monkeypatch.setattr(preview, "preview_registered_factories", lambda *args, **kwargs: object())
    plan = build_plan(ROOT, tmp_path / "runs", tmp_path, valid_hours=168)
    envelopes = sign_plan(ROOT, plan, key)
    started = []

    class FakeRunner:
        def __init__(self, output, cohort, replicate):
            self.output, self.cohort, self.replicate = output, cohort, replicate

        def run_authorized_multi(self, scenarios, *, envelope, evidence_root):
            started.append((self.cohort, self.replicate))
            if len(started) == 2:
                raise RuntimeError("simulated interruption")
            directory = self.output / "first"
            directory.mkdir()
            model = directory / "fake.zip"
            model.write_bytes(b"fake")
            return SimpleNamespace(final_model_path=model, run_directory=directory, actual_total_timesteps=200704)

    monkeypatch.setattr(checkpoints, "prepare_v4_registered_runner", lambda root, output, **run: FakeRunner(output, **run))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        batch.execute_batch(ROOT, plan, envelopes)
    events = [json.loads(line) for line in (tmp_path / "runs" / "batch-journal.jsonl").read_text().splitlines()]
    assert len(started) == 2
    assert [event["event"] for event in events] == ["started", "completed", "started", "failed"]
    assert not (tmp_path / "runs" / "batch-result.json").exists()
