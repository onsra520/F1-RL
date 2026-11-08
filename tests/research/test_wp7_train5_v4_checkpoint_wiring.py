from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import f1rl.learning.training.runner as runner_module
import f1rl.research.wp7_train5_v4_checkpoint as checkpoint_module
from f1rl.learning.training import TrainingRunResult
from f1rl.research.wp7_train5_v4_budget import load_v4_budget
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_BY_REPLICATE, FROZEN_CHECKPOINTS, evaluate_v4_runs, prepare_v4_registered_runner, prepare_v4_runner
from f1rl.research.wp7_train5_v4_admission import development_seeds, load_v4_protocol
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from tests.learning.training.mocks.mock_factories import make_factory
from tests.learning.training.mocks.mock_ppo import MockPPOModel
from tests.learning.training.mocks.mock_vec_env import MockVecEnv
from tests.research.test_wp7_train5_v4_budget_and_metrics import LEGACY, PILOT, ROOT, rows


@pytest.mark.parametrize("step,success", [(200704, True), (200000, False)])
def test_final_only_runner_checks_actual_step(tmp_path, monkeypatch, step, success):
    MockPPOModel.reset()
    MockPPOModel.progress_points = (25008, 51200, step)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: MockVecEnv())
    runner = prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=456)
    if not success:
        with pytest.raises(ValueError, match="unexpected timestep"):
            runner.run_single(make_factory())
        assert not list(tmp_path.rglob("*-final-model.zip"))
        return
    result = runner.run_single(make_factory())
    assert result.requested_total_timesteps == 200000
    assert result.actual_total_timesteps == 200704
    assert result.checkpoint_paths == ()
    assert result.final_model_path.is_file()
    assert len(MockPPOModel.instances[-1].save_calls) == 1


def test_metrics_bind_to_the_six_saved_final_checkpoints(tmp_path):
    runs = {}
    draft, draft_sha256 = load_v4_protocol(ROOT)
    registered_seeds = development_seeds(ROOT, draft, draft_sha256)
    for cohort in ("scratch", "frozen_initialization"):
        for replicate in range(3):
            directory = tmp_path / f"{cohort}-{replicate}"
            directory.mkdir()
            archive = directory / "final.zip"
            archive.write_bytes(f"{cohort}-{replicate}".encode())
            state = {"status": "completed", "requested_total_timesteps": 200000,
                     "actual_total_timesteps": 200704, "checkpoint_every_steps": None,
                     "latest_checkpoint": None, "final_model": "final.zip",
                     "run_id": directory.name, "mode": "multitrack", "vector_env": PILOT["vector_env"]}
            seed_group = "train5_from_scratch" if cohort == "scratch" else "train5_from_frozen_checkpoint"
            allocation = next(row for row in draft["runs"] if row["cohort"] == seed_group and row["replicate"] == replicate)
            state.update(ppo={**PILOT["ppo"], "seed": allocation["seeds"]["ppo"]}, sampling_seed=allocation["seeds"]["sampling"],
                         training_reset_seed=allocation["seeds"]["perturbation"], training_tracks=draft["track_ids"])
            if cohort == "frozen_initialization":
                identity = FROZEN_BY_REPLICATE[replicate]
                state["initial_checkpoint"] = {"id": identity, "sha256": FROZEN_CHECKPOINTS[identity], "policy_only": True}
            (directory / "state.json").write_text(json.dumps(state), encoding="utf8")
            runs[(cohort, replicate)] = TrainingRunResult(
                directory.name, directory, archive, (), 200000, 200704,
            )
    records = rows()
    for row in records:
        archive = runs[(row["cohort"], row["replicate"])].final_model_path
        row["checkpoint_sha256"] = sha256(archive.read_bytes()).hexdigest()
        cohort = "train5_from_scratch" if row["cohort"] == "scratch" else "train5_from_frozen_checkpoint"
        row["evaluation_seed"] = registered_seeds[(cohort, row["replicate"], row["track_id"], row["condition"], row["episode_index"])]
    result = evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)
    assert result["final_checkpoint_step"] == 200704
    assert result["scientific_training_permission"] is False
    records[0]["evaluation_seed"] += 1
    with pytest.raises(ValueError, match="record seed"):
        evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)
    records[0]["evaluation_seed"] -= 1
    scratch_state_path = runs[("scratch", 0)].run_directory / "state.json"
    scratch_state = json.loads(scratch_state_path.read_text(encoding="utf8"))
    scratch_state["training_reset_seed"] += 1
    scratch_state_path.write_text(json.dumps(scratch_state), encoding="utf8")
    with pytest.raises(ValueError, match="run state"):
        evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)
    scratch_state["training_reset_seed"] -= 1
    scratch_state_path.write_text(json.dumps(scratch_state), encoding="utf8")
    scratch_state["ppo"]["gamma"] = 0.1
    scratch_state_path.write_text(json.dumps(scratch_state), encoding="utf8")
    with pytest.raises(ValueError, match="run state"):
        evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)
    scratch_state["ppo"]["gamma"] = PILOT["ppo"]["gamma"]
    scratch_state_path.write_text(json.dumps(scratch_state), encoding="utf8")
    state_path = runs[("frozen_initialization", 0)].run_directory / "state.json"
    state = json.loads(state_path.read_text(encoding="utf8"))
    state["initial_checkpoint"] = None
    state_path.write_text(json.dumps(state), encoding="utf8")
    with pytest.raises(ValueError, match="frozen initializer"):
        evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)
    state["initial_checkpoint"] = {"id": "C-seed0", "sha256": FROZEN_CHECKPOINTS["C-seed0"], "policy_only": True}
    state_path.write_text(json.dumps(state), encoding="utf8")
    runs[("scratch", 0)].final_model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        evaluate_v4_runs(records, runs, LEGACY, load_v4_budget(ROOT), PILOT)


def test_metrics_reject_intermediate_checkpoint(tmp_path):
    archive = tmp_path / "model.zip"
    archive.write_bytes(b"model")
    run = TrainingRunResult("test", tmp_path, archive, (), 200000, 200704)
    runs = {(cohort, i): run for cohort in ("scratch", "frozen_initialization") for i in range(3)}
    runs[("scratch", 0)] = replace(run, checkpoint_paths=(archive,))
    with pytest.raises(ValueError, match="final-checkpoint contract"):
        evaluate_v4_runs([], runs, LEGACY, load_v4_budget(ROOT), PILOT)


def test_runner_rejects_artifacts_inside_source_checkout():
    with pytest.raises(ValueError, match="outside the source checkout"):
        prepare_v4_runner(ROOT, ROOT / "artifacts", ppo_seed=123, sampling_seed=456)


def test_runner_preserves_research_admission_guard(tmp_path):
    factory = replace(make_factory(), body_geometry=VehicleBodyGeometry(1.6, 1.2, 4.0, Vec3(0, .5, 0)))
    runner = prepare_v4_runner(ROOT, tmp_path / "runs", ppo_seed=123, sampling_seed=456)
    with pytest.raises(ValueError, match="training is blocked"):
        runner.run_single(factory)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("checkpoint", tuple(FROZEN_CHECKPOINTS))
def test_frozen_initialization_uses_verified_bytes_and_preserves_source(tmp_path, monkeypatch, checkpoint):
    archive = ROOT / "models" / checkpoint / "model-final.zip"
    before = archive.read_bytes()
    loaded = []
    MockPPOModel.reset()
    MockPPOModel.progress_points = (200704,)
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: MockVecEnv())
    monkeypatch.setattr(checkpoint_module, "initialize_policy_from_archive", lambda model, data: loaded.append(data))
    runner = prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=456, frozen_checkpoint=checkpoint)
    result = runner.run_single(make_factory())
    assert loaded == [before]
    assert sha256(before).hexdigest() == FROZEN_CHECKPOINTS[checkpoint]
    assert archive.read_bytes() == before
    assert result.final_model_path != archive
    assert result.actual_total_timesteps == 200704
    state = json.loads((result.run_directory / "state.json").read_text(encoding="utf8"))
    assert state["initial_checkpoint"] == {"id": checkpoint, "sha256": FROZEN_CHECKPOINTS[checkpoint], "policy_only": True}


@pytest.mark.parametrize("failure", ["missing", "wrong_hash"])
def test_invalid_frozen_checkpoint_rejected_before_allocation(tmp_path, monkeypatch, failure):
    archive = ROOT / "models" / "C-seed0" / "model-final.zip"
    original = Path.read_bytes

    def read_bytes(path):
        if path == archive:
            if failure == "missing":
                raise FileNotFoundError(str(path))
            return b"wrong checkpoint"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    destination = tmp_path / "not-created"
    with pytest.raises(ValueError, match="frozen checkpoint|protocol source"):
        prepare_v4_runner(ROOT, destination, ppo_seed=123, sampling_seed=456, frozen_checkpoint="C-seed0")
    assert not destination.exists()


def test_checkpoint_rechecked_before_model_initialization(tmp_path, monkeypatch):
    archive = ROOT / "models" / "C-seed0" / "model-final.zip"
    original = Path.read_bytes
    changed = False

    def read_bytes(path):
        if path == archive and changed:
            return b"changed after preparation"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    MockPPOModel.reset()
    monkeypatch.setattr(runner_module, "PPOModel", MockPPOModel)
    monkeypatch.setattr(runner_module, "make_vector_env", lambda *args: MockVecEnv())
    monkeypatch.setattr(checkpoint_module, "initialize_policy_from_archive", lambda model, data: pytest.fail("unverified checkpoint loaded"))
    runner = prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=456, frozen_checkpoint="C-seed0")
    changed = True
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.run_single(make_factory())
    assert MockPPOModel.instances[-1].learn_calls == []


def test_frozen_initialization_cannot_bypass_research_guard(tmp_path):
    factory = replace(make_factory(), body_geometry=VehicleBodyGeometry(1.6, 1.2, 4.0, Vec3(0, .5, 0)))
    runner = prepare_v4_runner(ROOT, tmp_path / "runs", ppo_seed=123, sampling_seed=456, frozen_checkpoint="C-seed0")
    with pytest.raises(ValueError, match="training is blocked"):
        runner.run_single(factory)
    assert not (tmp_path / "runs").exists()


def test_unknown_frozen_checkpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unregistered"):
        prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=456, frozen_checkpoint="A-seed0")


def test_pilot_configuration_uses_only_verified_snapshot(tmp_path, monkeypatch):
    pilot = ROOT / "configs/recovery/wp7-pilot-200k.yml"
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == pilot:
            return original(path, *args, **kwargs).replace("gamma: 0.99", "gamma: 0.10")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    runner = prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=456)
    assert runner._config.ppo.gamma == 0.99


def test_registered_runner_uses_fixed_seeds_and_rejects_wrong_roster(tmp_path):
    from tests.learning.training.mocks.mock_factories import make_scenario_set

    runner = prepare_v4_registered_runner(ROOT, tmp_path, cohort="train5_from_frozen_checkpoint", replicate=0)
    draft, _ = load_v4_protocol(ROOT)
    seeds = draft["runs"][3]["seeds"]
    assert runner._config.ppo.seed == seeds["ppo"]
    assert runner._config.sampling_seed == seeds["sampling"]
    assert runner._registered_run["perturbation_seed"] == seeds["perturbation"]
    assert runner._initial_checkpoint["id"] == "C-seed0"
    with pytest.raises(ValueError, match="track roster"):
        runner.run_multi(make_scenario_set())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("ppo_seed,sampling_seed", [(2026111001, 123), (2026111000, 123), (2026111008, 123), (123, 2026112101)])
def test_unregistered_runner_refuses_reserved_seeds(tmp_path, ppo_seed, sampling_seed):
    with pytest.raises(ValueError, match="reserved TRAIN-5 seeds"):
        prepare_v4_runner(ROOT, tmp_path, ppo_seed=ppo_seed, sampling_seed=sampling_seed)
    assert not list(tmp_path.iterdir())


def test_unregistered_runner_rejects_vector_seed_overflow(tmp_path):
    with pytest.raises(ValueError, match="vector ranks"):
        prepare_v4_runner(ROOT, tmp_path, ppo_seed=2**32 - 1, sampling_seed=123)


def test_unregistered_runner_refuses_registered_development_seed(tmp_path):
    draft, digest = load_v4_protocol(ROOT)
    seed = next(iter(development_seeds(ROOT, draft, digest).values()))
    with pytest.raises(ValueError, match="reserved TRAIN-5 seeds"):
        prepare_v4_runner(ROOT, tmp_path, ppo_seed=seed - 1, sampling_seed=123)
    with pytest.raises(ValueError, match="reserved TRAIN-5 seeds"):
        prepare_v4_runner(ROOT, tmp_path, ppo_seed=123, sampling_seed=seed)


def test_registered_runner_rejects_wrong_reset_seed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import f1rl.research.wp7_train5_v4_runner as v4_runner

    runner = prepare_v4_registered_runner(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)
    class ScenarioSet:
        scenarios = tuple(
            SimpleNamespace(track_id=track, factory=SimpleNamespace(
                reset_track_id=track, reset_config=SimpleNamespace(seed=0, version="wp7_train_reset_lifecycle_research_v1", max_attempts=32),
            )) for track in runner._registered_run["track_ids"]
        )
    monkeypatch.setattr(v4_runner, "TrainingScenarioSet", ScenarioSet)
    with pytest.raises(ValueError, match="reset configuration"):
        runner.run_multi(ScenarioSet())
    assert not list(tmp_path.iterdir())


def test_registered_runner_denies_even_when_ids_and_reset_labels_match(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import f1rl.research.wp7_train5_v4_runner as v4_runner

    runner = prepare_v4_registered_runner(ROOT, tmp_path, cohort="train5_from_scratch", replicate=0)
    class ScenarioSet:
        scenarios = tuple(
            SimpleNamespace(track_id=track, factory=SimpleNamespace(
                reset_track_id=track, reset_config=SimpleNamespace(
                    seed=runner._registered_run["perturbation_seed"],
                    version="wp7_train_reset_lifecycle_research_v1", max_attempts=32),
            )) for track in runner._registered_run["track_ids"]
        )
    monkeypatch.setattr(v4_runner, "TrainingScenarioSet", ScenarioSet)
    with pytest.raises(ValueError, match="admission is not granted"):
        runner.run_multi(ScenarioSet())
    assert not list(tmp_path.iterdir())
