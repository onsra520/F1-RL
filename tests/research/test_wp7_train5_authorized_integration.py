from datetime import datetime, timedelta, timezone
from dataclasses import replace
from hashlib import sha256
import json
import subprocess
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import numpy as np
import pytest

from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel, _validate_training_admission
from f1rl.learning.training import VectorEnvConfig, make_multi_track_vector_env
from f1rl.learning.training import runner as core_runner
from f1rl.research.wp7_train5_v4_admission import load_v4_protocol
from f1rl.research.wp7_train5_v4_checkpoint import prepare_v4_registered_runner
import f1rl.research.wp7_train5_v4_checkpoint as checkpoint
from scripts.wp7_train5_execution_authorization import authorization_message
from scripts.wp7_train5_v4_review_manifest import load_review_manifest
from tests.learning.training.mocks.mock_ppo import MockPPOModel
import scripts.wp7_train5_v4_factory_preview as preview

ROOT = Path(__file__).resolve().parents[2]


def make_authorized_fixture(tmp_path, monkeypatch, cohort="train5_from_scratch", replicate=0):
    from f1rl.research import wp7_train5_internal_execution as execution

    monkeypatch.setattr(preview, "audit", lambda *_: {"external_witness_status": "FOUR_EXTERNAL_WITNESSES_SHA256_VERIFIED"})
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(execution, "_owner_public_key", lambda root, expected: key.public_key().public_bytes_raw())
    _, protocol_sha = load_v4_protocol(ROOT)
    _, manifest_sha = load_review_manifest(ROOT)
    now = datetime.now(timezone.utc)
    output = (tmp_path / "runs").resolve()
    payload = {"schema": "train5_local_execution_review_v1", "purpose": "simulation_only_train5",
               "source_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                capture_output=True, text=True, check=True).stdout.strip(),
               "protocol_sha256": protocol_sha, "manifest_sha256": manifest_sha,
               "cohort": cohort, "replicate": replicate, "output_root": str(output),
               "nonce": "2" * 32, "issued_at_utc": (now - timedelta(minutes=1)).isoformat(),
               "expires_at_utc": (now + timedelta(minutes=10)).isoformat(),
               "accept_missing_seed_history": True, "accept_missing_c2_replay": True,
               "accept_internal_review_pending": True}
    envelope = {"payload": payload, "signature_hex": key.sign(authorization_message(payload)).hex()}
    scenarios = preview.preview_registered_factories(ROOT, tmp_path, cohort=cohort, replicate=replicate)

    class GuardedFakeModel(MockPPOModel):
        @classmethod
        def create(cls, env, config):
            from f1rl.research import wp7_train5_internal_execution as execution
            assert env.num_envs == 8
            assert execution._active.get()["max_steps"] == 200704
            _validate_training_admission(env)
            return super().create(env, config)

        def learn(self, total_timesteps, **kwargs):
            from f1rl.research import wp7_train5_internal_execution as execution
            _validate_training_admission(self.env)
            execution._active.get()["steps"] = 200704
            return super().learn(total_timesteps, **kwargs)

    GuardedFakeModel.reset()
    GuardedFakeModel.progress_points = (200704,)
    monkeypatch.setattr(core_runner, "PPOModel", GuardedFakeModel)
    runner = prepare_v4_registered_runner(ROOT, output, cohort=cohort, replicate=replicate)
    return runner, scenarios, envelope, tmp_path, output, GuardedFakeModel


@pytest.fixture
def authorized_fixture(tmp_path, monkeypatch):
    return make_authorized_fixture(tmp_path, monkeypatch)


def test_fake_signed_run_crosses_eight_selector_and_ppo_guards(authorized_fixture):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    result = runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert result.actual_total_timesteps == 200704
    assert result.final_model_path.is_file()
    assert len(model.instances) == 1
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf8"))
    assert state["execution_scope"] == "INTERNAL_SIMULATION_ONLY"
    assert state["internal_authorization"] == envelope


def test_fake_model_resets_and_steps_all_eight_selectors(authorized_fixture, monkeypatch):
    from f1rl.learning.gym_environment import F1RLEnv
    from f1rl.research import wp7_train5_internal_execution as execution

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    monkeypatch.setattr(F1RLEnv, "reset", lambda self, **kwargs: (
        np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}))
    monkeypatch.setattr(F1RLEnv, "step", lambda self, action: (
        np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), 0.0, False, False, {}))

    class SteppingFakeModel(model):
        def learn(self, total_timesteps, **kwargs):
            _validate_training_admission(self.env)
            observations = self.env.reset()
            assert observations.shape[0] == 8
            assert all(info["track_id"] in preview.TRAIN5_IDS for info in self.env.reset_infos)
            self.env.step(np.zeros((8, *self.env.action_space.shape), dtype=self.env.action_space.dtype))
            assert execution._active.get()["steps"] == 8
            execution._active.get()["steps"] = 200704
            return MockPPOModel.learn(self, total_timesteps, **kwargs)

    monkeypatch.setattr(core_runner, "PPOModel", SteppingFakeModel)
    result = runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert result.actual_total_timesteps == 200704


@pytest.mark.parametrize("replicate", [1, 2])
def test_other_fake_scratch_replicates_use_registered_scope(tmp_path, monkeypatch, replicate):
    runner, scenarios, envelope, evidence, output, model = make_authorized_fixture(
        tmp_path, monkeypatch, "train5_from_scratch", replicate)
    result = runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert result.actual_total_timesteps == 200704
    assert len(model.instances) == 1


@pytest.mark.parametrize("replicate", [0, 1, 2])
def test_fake_frozen_runs_load_registered_checkpoint_and_preserve_original(tmp_path, monkeypatch, replicate):
    identity = checkpoint.FROZEN_BY_REPLICATE[replicate]
    source = ROOT / "models" / identity / "model-final.zip"
    before = sha256(source.read_bytes()).hexdigest()
    loaded = []
    monkeypatch.setattr(checkpoint, "initialize_policy_from_archive", lambda model, data: loaded.append(sha256(data).hexdigest()))
    runner, scenarios, envelope, evidence, output, model = make_authorized_fixture(
        tmp_path, monkeypatch, "train5_from_frozen_checkpoint", replicate)
    result = runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert loaded == [checkpoint.FROZEN_CHECKPOINTS[identity]]
    assert sha256(source.read_bytes()).hexdigest() == before
    assert result.final_model_path != source


def test_signed_run_cannot_be_replayed(authorized_fixture):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    with pytest.raises(FileExistsError):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert len(model.instances) == 1


def test_capability_is_revoked_after_fake_run(authorized_fixture):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    env = make_multi_track_vector_env(scenarios, VectorEnvConfig(n_envs=8), sampling_seed=0)
    try:
        with pytest.raises(ValueError, match="training is blocked"):
            PPOModel.create(env, PPOConfig())
    finally:
        env.close()


@pytest.mark.parametrize("failure", ["count", "rank", "rng"])
def test_modified_eight_selector_binding_is_rejected(authorized_fixture, monkeypatch, failure):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    original = core_runner.make_multi_track_vector_env

    def altered(scenarios, config, sampling_seed):
        if failure == "count":
            config = replace(config, n_envs=7)
        env = original(scenarios, config, sampling_seed)
        if failure == "rank":
            env.envs[3]._rank = 99
        elif failure == "rng":
            env.envs[3]._selection_rng.random()
        return env

    monkeypatch.setattr(core_runner, "make_multi_track_vector_env", altered)
    with pytest.raises(ValueError, match="not admitted|fresh and unmodified"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not list(output.rglob("*-final-model.zip"))


def test_wrong_signature_rejected_before_allocation(authorized_fixture):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    envelope["signature_hex"] = "00" * 64
    with pytest.raises(ValueError, match="signature"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists()


def test_changed_output_root_rejected_before_allocation(authorized_fixture):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    different = evidence / "different-output"
    runner._config = replace(runner._config, artifacts_root=different)
    with pytest.raises(ValueError, match="scope differs"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists() and not different.exists()


def test_modified_scenario_rejected_before_allocation(authorized_fixture):
    from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    scenarios = TrainingScenarioSet(tuple(TrainingScenario(row.track_id, replace(
        row.factory, reset_config=replace(row.factory.reset_config, seed=0))) for row in scenarios.scenarios))
    with pytest.raises(ValueError, match="scenario geometry"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists()


def test_missing_track_rejected_before_allocation(authorized_fixture):
    from f1rl.learning.training.env_factory import TrainingScenarioSet

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    subset = TrainingScenarioSet(scenarios.scenarios[:-1])
    with pytest.raises(ValueError, match="scenario geometry"):
        runner.run_authorized_multi(subset, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists()


def test_expiry_stops_an_active_fake_run_before_stepping(authorized_fixture, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    actual_datetime = datetime

    class FutureClock:
        @staticmethod
        def now(tz):
            return actual_datetime.now(tz) + timedelta(hours=1)

    class ExpiringFakeModel(model):
        def learn(self, total_timesteps, **kwargs):
            monkeypatch.setattr(execution, "datetime", FutureClock)
            self.env.envs[0].step(None)
            pytest.fail("expired authorization permitted a step")

    monkeypatch.setattr(core_runner, "PPOModel", ExpiringFakeModel)
    with pytest.raises(ValueError, match="expired during execution"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not list(output.rglob("*-final-model.zip"))


def test_active_step_budget_stops_before_an_extra_step(authorized_fixture, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution

    runner, scenarios, envelope, evidence, output, model = authorized_fixture

    class BudgetFakeModel(model):
        def learn(self, total_timesteps, **kwargs):
            execution._active.get()["max_steps"] = 1
            execution.check_active_step(self.env.envs[0])
            execution.check_active_step(self.env.envs[1])
            pytest.fail("step budget was exceeded")

    monkeypatch.setattr(core_runner, "PPOModel", BudgetFakeModel)
    with pytest.raises(ValueError, match="step budget exhausted"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not list(output.rglob("*-final-model.zip"))


def test_false_model_step_count_cannot_produce_a_final_checkpoint(authorized_fixture, monkeypatch):
    runner, scenarios, envelope, evidence, output, model = authorized_fixture

    class LyingFakeModel(model):
        def learn(self, total_timesteps, **kwargs):
            return MockPPOModel.learn(self, total_timesteps, **kwargs)

    monkeypatch.setattr(core_runner, "PPOModel", LyingFakeModel)
    with pytest.raises(ValueError, match="model and environment steps differ"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not list(output.rglob("*-final-model.zip"))


def test_missing_owner_key_rejected_before_allocation(authorized_fixture, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    monkeypatch.undo()
    original = Path.read_bytes

    def missing(path):
        if path == ROOT / execution.OWNER_KEY:
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", missing)
    with pytest.raises((FileNotFoundError, ValueError)):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists()


def test_owner_registry_change_after_manifest_check_is_rejected(authorized_fixture, monkeypatch):
    from f1rl.research import wp7_train5_internal_execution as execution

    runner, scenarios, envelope, evidence, output, model = authorized_fixture
    monkeypatch.undo()
    path = ROOT / execution.OWNER_KEY
    original = Path.read_bytes
    reads = 0

    def changed(path_obj):
        nonlocal reads
        if path_obj == path:
            reads += 1
            if reads > 1:
                return b'{}'
        return original(path_obj)

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="owner key registry changed"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert reads == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["missing", "wrong_hash"])
def test_frozen_checkpoint_failure_rejected_before_allocation(tmp_path, monkeypatch, failure):
    runner, scenarios, envelope, evidence, output, model = make_authorized_fixture(
        tmp_path, monkeypatch, "train5_from_frozen_checkpoint", 0)
    source = ROOT / "models" / "C-seed0" / "model-final.zip"
    original = Path.read_bytes

    def read_bytes(path):
        if path == source:
            if failure == "missing":
                raise FileNotFoundError(path)
            return b"invalid checkpoint"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(ValueError, match="source"):
        runner.run_authorized_multi(scenarios, envelope=envelope, evidence_root=evidence)
    assert not model.instances
    assert not output.exists()
