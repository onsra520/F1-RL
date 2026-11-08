from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from scripts.wp7_train5_execution_authorization import authorization_message, verify_authorization
from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
from f1rl.learning.training import VectorEnvConfig, make_multi_track_vector_env
from f1rl.learning.training import runner as core_runner
from f1rl.research.wp7_train5_v4_checkpoint import prepare_v4_registered_runner
from tests.learning.training.mocks.mock_ppo import MockPPOModel
import scripts.wp7_train5_v4_factory_preview as preview

ROOT = Path(__file__).resolve().parents[2]


def signed_fixture():
    key = Ed25519PrivateKey.generate()
    now = datetime(2026, 11, 9, 10, tzinfo=timezone.utc)
    payload = {
        "schema": "train5_local_execution_review_v1",
        "purpose": "simulation_only_train5",
        "protocol_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "cohort": "train5_from_scratch",
        "replicate": 0,
        "output_root": "D:/fixture-output",
        "nonce": "1" * 32,
        "issued_at_utc": "2026-11-09T09:00:00+00:00",
        "expires_at_utc": "2026-11-09T11:00:00+00:00",
        "accept_missing_seed_history": True,
        "accept_missing_c2_replay": True,
    }
    envelope = {"payload": payload, "signature_hex": key.sign(authorization_message(payload)).hex()}
    return envelope, key.public_key().public_bytes_raw(), now


def test_scoped_fake_owner_signature_is_verified_without_granting_training():
    envelope, public_key, now = signed_fixture()
    result = verify_authorization(envelope, public_key,
                                  protocol_sha256="a" * 64, manifest_sha256="b" * 64,
                                  cohort="train5_from_scratch", replicate=0,
                                  output_root="D:/fixture-output", now=now)
    assert result["signature_verified"] is True
    assert result["training_permitted"] is False


@pytest.mark.parametrize("failure", ["signature", "protocol", "manifest", "replicate", "expiry", "output", "nonce", "seed_exception", "c2_exception"])
def test_forged_expired_or_wrong_scope_signature_is_rejected(failure):
    envelope, public_key, now = signed_fixture()
    payload = envelope["payload"]
    if failure == "signature":
        envelope["signature_hex"] = "00" * 64
    elif failure == "protocol":
        payload["protocol_sha256"] = "c" * 64
    elif failure == "manifest":
        payload["manifest_sha256"] = "c" * 64
    elif failure == "replicate":
        payload["replicate"] = 1
    elif failure == "expiry":
        now = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
    elif failure == "output":
        payload["output_root"] = "D:/different-output"
    elif failure == "nonce":
        payload["nonce"] = "../bad"
    elif failure == "seed_exception":
        payload["accept_missing_seed_history"] = False
    else:
        payload["accept_missing_c2_replay"] = False
    with pytest.raises(ValueError):
        verify_authorization(envelope, public_key, protocol_sha256="a" * 64,
                             manifest_sha256="b" * 64, cohort="train5_from_scratch",
                             replicate=0, output_root="D:/fixture-output", now=now)


def test_signed_fixture_does_not_unlock_eight_real_track_selectors(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"})
    scenarios = preview.preview_registered_factories(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)
    envelope, public_key, now = signed_fixture()
    receipt = verify_authorization(envelope, public_key, protocol_sha256="a" * 64,
                                   manifest_sha256="b" * 64, cohort="train5_from_scratch", replicate=0,
                                   output_root="D:/fixture-output", now=now)
    assert receipt["training_permitted"] is False
    env = make_multi_track_vector_env(scenarios, VectorEnvConfig(n_envs=8), sampling_seed=0)
    try:
        assert env.num_envs == 8
        assert env.get_attr("requires_scientific_admission") == [True] * 8
        with pytest.raises(ValueError, match="training is blocked"):
            PPOModel.create(env, PPOConfig())
    finally:
        env.close()


def test_runner_still_denies_when_only_untrusted_signature_was_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"})
    scenarios = preview.preview_registered_factories(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)
    envelope, public_key, now = signed_fixture()
    verify_authorization(envelope, public_key, protocol_sha256="a" * 64,
                         manifest_sha256="b" * 64, cohort="train5_from_scratch", replicate=0,
                         output_root="D:/fixture-output", now=now)
    MockPPOModel.reset()
    monkeypatch.setattr(core_runner, "PPOModel", MockPPOModel)
    destination = tmp_path / "runs"
    runner = prepare_v4_registered_runner(ROOT, destination, cohort="train5_from_scratch", replicate=0)
    with pytest.raises(ValueError, match="admission is not granted"):
        runner.run_multi(scenarios)
    assert not MockPPOModel.instances
    assert not destination.exists()
