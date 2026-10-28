"""Synthetic TRAIN registry and resets; no policy learning or heldout access."""
from dataclasses import replace

import numpy as np
import pytest

from src.learning.train_reset import TrainResetConfig, SpawnFeasibilityError
from src.track_assets.canonical import fingerprint_track_world
from src.track_assets.model import TrackManifestEntry
from tests.integration.test_wp7_race_contract import race_factory


def randomized_factory(seed=13, *, track=None):
    candidate = race_factory(checkpoints=(), finish=50., speed=0.)
    if track is not None:
        candidate = replace(candidate, track=track)
    entry = TrackManifestEntry("synthetic-train", "training/synthetic.json",
                               fingerprint_track_world(candidate.track), "0"*64)
    config = TrainResetConfig(seed, (entry,))
    return replace(candidate, reset_config=config, reset_track_id=entry.track_id)


def test_repeated_seeded_gym_resets_use_independent_replayable_perturbation_stream():
    candidate = randomized_factory()
    a, b = candidate.create(), candidate.create()
    try:
        first = None
        for index in range(3):
            obs_a, info_a = a.reset(seed=999)
            obs_b, info_b = b.reset(seed=12)
            assert np.array_equal(obs_a, obs_b)
            assert info_a["train_reset"] == info_b["train_reset"]
            sample = info_a["train_reset"]
            assert sample["reset_index"] == index
            assert abs(sample["lateral_offset_m"]) <= .5
            assert abs(sample["heading_jitter_deg"]) <= 5.
            assert info_a["telemetry"]["vehicle_position"][0] == pytest.approx(sample["lateral_offset_m"])
            assert sample["world_sha256"] == fingerprint_track_world(candidate.track)
            if first is None:
                first = sample["lateral_offset_m"]
            else:
                assert sample["lateral_offset_m"] != first
            info = a.step(np.array([0., 0.], dtype=np.float32))[-1]
            assert info["train_reset"] == sample
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("sign", [-1., 1.])
def test_registered_spawn_range_endpoints_are_feasible(monkeypatch, sign):
    import src.learning.train_reset as reset_module
    class EndpointRandom:
        def __init__(self, seed):
            pass
        def uniform(self, low, high):
            return low if sign < 0. else high
    monkeypatch.setattr(reset_module.random, "Random", EndpointRandom)
    env = randomized_factory().create()
    try:
        info = env.reset()[1]
        assert info["train_reset"]["lateral_offset_m"] == sign*.5
        assert info["train_reset"]["heading_jitter_deg"] == sign*5.
        assert env.step(np.array([0., 0.], dtype=np.float32))[-1]["tick"] == 1
    finally:
        env.close()


def test_infeasible_draw_is_logged_and_redrawn_without_nominal_fallback(monkeypatch):
    import src.learning.train_reset as reset_module
    from src.sandbox.entities.obstacle import Obstacle
    from tests.integration.test_wp7_race_contract import Vec3, Quaternion, OrientedBoxVolume
    class Draws:
        def __init__(self, seed):
            self.values = iter((.5, 0., -.5, 0.))
        def uniform(self, low, high):
            return next(self.values)
    monkeypatch.setattr(reset_module.random, "Random", Draws)
    base = race_factory(checkpoints=(), finish=50., speed=0.)
    obstacle = Obstacle("side", OrientedBoxVolume(
        Vec3(1.2, .8, 0.), Vec3(.2, .8, .3), Quaternion.identity(),
    ))
    env = randomized_factory(track=replace(base.track, obstacles=(obstacle,))).create()
    try:
        sample = env.reset()[1]["train_reset"]
        assert sample["attempt"] == 1
        assert sample["lateral_offset_m"] == -.5
        assert sample["rejected_draws"][0]["reason"] == "body_obstacle"
    finally:
        env.close()


def test_no_feasible_spawn_rejects_reset_and_cannot_step():
    from src.sandbox.entities.obstacle import Obstacle
    from tests.integration.test_wp7_race_contract import Vec3, Quaternion, OrientedBoxVolume
    base = race_factory(checkpoints=(), finish=50., speed=0.)
    wall = Obstacle("wall", OrientedBoxVolume(
        Vec3(0., .8, 0.), Vec3(.8, .8, .3), Quaternion.identity(),
    ))
    candidate = randomized_factory(track=replace(base.track, obstacles=(wall,)))
    env = replace(candidate, reset_config=replace(candidate.reset_config, max_attempts=2)).create()
    try:
        with pytest.raises(SpawnFeasibilityError, match="no feasible TRAIN spawn in 2 draws"):
            env.reset()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.array([0., 0.], dtype=np.float32))
    finally:
        env.close()


def test_heldout_metadata_and_track_identity_drift_are_rejected():
    candidate = randomized_factory()
    entry = candidate.reset_config.training_registry[0]
    with pytest.raises(ValueError, match="HELD-OUT/non-training"):
        TrainResetConfig(1, (replace(entry, path="evaluation/sealed.json"),))
    with pytest.raises(ValueError, match="identity is absent or mismatched"):
        replace(candidate, reset_track_id="unregistered")
    with pytest.raises(ValueError, match="identity is absent or mismatched"):
        replace(candidate, track=replace(candidate.track, surfaces=(
            replace(candidate.track.surfaces[0], width=30.),)))
    from src.learning.training.env_factory import TrainingScenario
    with pytest.raises(ValueError, match="scenario ID"):
        TrainingScenario("other", candidate)


@pytest.mark.parametrize("backend", ["dummy", "subproc"])
def test_vector_rank_and_reset_sequence_survive_environment_recreation(backend):
    from src.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
    from src.learning.training.vector_env import make_multi_track_vector_env
    from src.learning.training.config import VectorEnvConfig
    candidate = randomized_factory()
    scenarios = TrainingScenarioSet((TrainingScenario(candidate.reset_track_id, candidate),))
    vec = make_multi_track_vector_env(scenarios, VectorEnvConfig(n_envs=2, backend=backend), 23)
    try:
        samples = []
        for index in range(2):
            vec.seed(99)
            assert vec.reset().shape == (2, 9)
            info = vec.step(np.zeros((2, 2), dtype=np.float32))[-1]
            pair = [row["train_reset"] for row in info]
            assert [row["rank"] for row in pair] == [0, 1]
            assert all(row["reset_index"] == index for row in pair)
            assert pair[0]["draw_seed"] != pair[1]["draw_seed"]
            samples.append(pair[0]["draw_seed"])
        assert samples[0] != samples[1]
    finally:
        vec.close()


def test_reset_preparation_does_not_open_training_admission(tmp_path, monkeypatch):
    from src.learning.algorithms.ppo import PPOConfig
    from src.learning.training.config import TrainingRunConfig, VectorEnvConfig
    from src.learning.training.runner import TrainingRunner
    def forbidden(*args, **kwargs):
        raise AssertionError("admission bypass")
    monkeypatch.setattr(TrainingRunner, "_run", forbidden)
    root = tmp_path / "forbidden"
    runner = TrainingRunner(TrainingRunConfig(PPOConfig(), VectorEnvConfig(n_envs=1), artifacts_root=root))
    with pytest.raises(ValueError, match="training is blocked"):
        runner.run_single(randomized_factory())
    assert not root.exists()
