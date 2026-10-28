import numpy as np
import pytest

from f1rl.learning import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
from f1rl.learning.gym_environment import F1RLEnv
from f1rl.runtime.simulation import SimulationConfig
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.perception.sensor_suite import SensorSuite
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


def _env() -> F1RLEnv:
    track = TrackWorld(
        finish_lines=(FinishLine(
            "finish",
            OrientedBoxVolume(Vec3(100.0, 0.0, 0.0), Vec3(0.5, 5.0, 5.0), Quaternion.identity()),
        ),)
    )
    state = WorldState(
        vehicle=VehicleState(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), False),
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(),
    )
    parameters = VehicleParameters(
        wheelbase=2.5,
        ride_height=0.3,
        max_steering_angle=0.6,
        max_acceleration=1.0,
        max_braking=0.0,
        rolling_drag=0.0,
        air_drag=0.0,
        max_ground_speed=50.0,
        gravity=9.81,
    )
    encoder = ObservationEncoder(
        ObservationConfig(
            sensor_suite=SensorSuite(()),
            velocity_scale=20.0,
            target_distance_scale=100.0,
        )
    )
    return F1RLEnv(
        track=track,
        initial_state=state,
        vehicle_parameters=parameters,
        simulation_config=SimulationConfig(0.1, 0.5),
        observation_encoder=encoder,
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(100),
    )


def _config(device: str = "cpu") -> PPOConfig:
    return PPOConfig(n_steps=8, batch_size=4, n_epochs=1, device=device)


def test_create_requires_concrete_environment_and_config() -> None:
    env = _env()
    with pytest.raises(ValueError):
        PPOModel.create(object(), _config())
    with pytest.raises(ValueError):
        PPOModel.create(env, object())


def test_create_forwards_baseline_configuration_to_sb3() -> None:
    env = _env()
    config = PPOConfig(
        learning_rate=1e-4,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.1,
        ent_coef=0.02,
        vf_coef=0.7,
        max_grad_norm=0.4,
        seed=7,
        device="cpu",
    )

    model = PPOModel.create(env, config)
    backend = model._model

    assert backend.policy.__class__.__name__ == "ActorCriticPolicy"
    assert backend.learning_rate == pytest.approx(config.learning_rate)
    assert backend.n_steps == config.n_steps
    assert backend.batch_size == config.batch_size
    assert backend.n_epochs == config.n_epochs
    assert backend.gamma == pytest.approx(config.gamma)
    assert backend.gae_lambda == pytest.approx(config.gae_lambda)
    assert backend.clip_range(1.0) == pytest.approx(config.clip_range)
    assert backend.ent_coef == pytest.approx(config.ent_coef)
    assert backend.vf_coef == pytest.approx(config.vf_coef)
    assert backend.max_grad_norm == pytest.approx(config.max_grad_norm)
    assert backend.seed == config.seed
    assert backend.device.type == "cpu"
    assert backend.verbose == 0
    assert model.num_timesteps == 0


def test_create_accepts_auto_device() -> None:
    model = PPOModel.create(_env(), _config("auto"))
    assert model._model.device.type in {"cpu", "cuda"}


def test_explicit_cuda_must_not_silently_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    import f1rl.learning.algorithms.ppo as ppo_module

    class FakePPO:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.device = type("Device", (), {"type": "cpu"})()
            self.num_timesteps = 0

    monkeypatch.setattr(ppo_module, "PPO", FakePPO)
    with pytest.raises(RuntimeError, match="CUDA"):
        PPOModel.create(_env(), _config("cuda"))


@pytest.mark.parametrize(
    "observation",
    [
        [0.0] * 9,
        np.zeros(8, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.zeros(9, dtype=np.bool_),
        np.zeros(9, dtype=object),
        np.array(["0"] * 9),
    ],
)
def test_predict_rejects_invalid_observation_type_shape_or_dtype(observation: object) -> None:
    model = PPOModel.create(_env(), _config())
    with pytest.raises(ValueError):
        model.predict(observation)


def test_predict_rejects_non_finite_observation() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    observation = observation.copy()
    observation[0] = np.nan
    with pytest.raises(ValueError):
        model.predict(observation)


def test_predict_rejects_observation_outside_space() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    observation = observation.copy()
    observation[0] = 2.0
    with pytest.raises(ValueError):
        model.predict(observation)


@pytest.mark.parametrize("deterministic", [0, 1, "true", None])
def test_predict_requires_boolean_deterministic(deterministic: object) -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    with pytest.raises(ValueError):
        model.predict(observation, deterministic=deterministic)


def test_predict_returns_valid_float32_action() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()

    action = model.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert env.action_space.contains(action)


@pytest.mark.parametrize("total_timesteps", [True, False, 0, -1, 1.5, "8", None])
def test_learn_rejects_invalid_total_timesteps_before_backend(total_timesteps: object) -> None:
    class Backend:
        num_timesteps = 0

        def get_env(self) -> object:
            return object()

        def learn(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("backend learn must not be called")

    model = PPOModel(Backend())
    with pytest.raises(ValueError):
        model.learn(total_timesteps)


def test_learn_uses_cumulative_backend_semantics() -> None:
    class Backend:
        num_timesteps = 3

        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def get_env(self) -> object:
            return _env()

        def learn(self, *, total_timesteps: int, reset_num_timesteps: bool) -> None:
            self.calls.append((total_timesteps, reset_num_timesteps))
            self.num_timesteps += total_timesteps

    backend = Backend()
    model = PPOModel(backend)

    assert model.learn(5) is model
    assert backend.calls == [(5, False)]
    assert model.num_timesteps == 8


def test_save_normalizes_zip_suffix_and_returns_archive(tmp_path) -> None:
    model = PPOModel.create(_env(), _config())
    destination = tmp_path / "policy.bin"

    saved = model.save(destination)

    assert saved == tmp_path / "policy.bin.zip"
    assert saved.is_file()


@pytest.mark.parametrize("path", ["", "   ", 123, None, object()])
def test_save_rejects_invalid_path_values(path: object) -> None:
    model = PPOModel.create(_env(), _config())
    with pytest.raises(ValueError):
        model.save(path)


def test_save_rejects_missing_parent(tmp_path) -> None:
    model = PPOModel.create(_env(), _config())
    with pytest.raises(ValueError):
        model.save(tmp_path / "missing" / "policy")


def test_save_rejects_directory_destination(tmp_path) -> None:
    model = PPOModel.create(_env(), _config())
    directory = tmp_path / "policy.zip"
    directory.mkdir()
    with pytest.raises(ValueError):
        model.save(directory)


@pytest.mark.parametrize("path", ["", "   ", 123, None, object()])
def test_load_rejects_invalid_path_values(path: object) -> None:
    with pytest.raises(ValueError):
        PPOModel.load(path)


def test_load_rejects_missing_archive(tmp_path) -> None:
    with pytest.raises(ValueError):
        PPOModel.load(tmp_path / "missing-policy")


def test_load_rejects_invalid_env_type(tmp_path) -> None:
    model = PPOModel.create(_env(), _config())
    path = model.save(tmp_path / "policy")
    with pytest.raises(ValueError):
        PPOModel.load(path, env=object())


@pytest.mark.parametrize("device", ["", "gpu", "CUDA", 1, None])
def test_load_rejects_invalid_device(device: object, tmp_path) -> None:
    model = PPOModel.create(_env(), _config())
    path = model.save(tmp_path / "policy")
    with pytest.raises(ValueError):
        PPOModel.load(path, device=device)


def test_inference_only_load_predicts_but_cannot_learn(tmp_path) -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    path = model.save(tmp_path / "policy")

    loaded = PPOModel.load(path, env=None, device="cpu")
    action = loaded.predict(observation)

    assert env.action_space.contains(action)
    with pytest.raises(RuntimeError, match="environment"):
        loaded.learn(8)


def test_load_cuda_must_not_silently_fall_back(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import f1rl.learning.algorithms.ppo as ppo_module

    archive = tmp_path / "policy.zip"
    archive.write_bytes(b"placeholder")

    class FakeLoaded:
        device = type("Device", (), {"type": "cpu"})()
        num_timesteps = 0

    class FakePPO:
        @classmethod
        def load(cls, *args: object, **kwargs: object) -> FakeLoaded:
            return FakeLoaded()

    monkeypatch.setattr(ppo_module, "PPO", FakePPO)
    with pytest.raises(RuntimeError, match="CUDA"):
        PPOModel.load(archive, device="cuda")


def test_single_env_binding_rejects_oversized_or_nondivisible_batch() -> None:
    oversized = PPOConfig(n_steps=8, batch_size=16, n_epochs=1)
    nondivisible = PPOConfig(n_steps=8, batch_size=3, n_epochs=1)

    with pytest.raises(ValueError):
        PPOModel.create(_env(), oversized)
    with pytest.raises(ValueError):
        PPOModel.create(_env(), nondivisible)


def test_vector_env_binding_uses_total_rollout_size() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    vector = DummyVecEnv([_env, _env, _env, _env])
    try:
        model = PPOModel.create(
            vector,
            PPOConfig(n_steps=8, batch_size=16, n_epochs=1),
        )
        assert model._model.n_envs == 4
        assert model._model.batch_size == 16
    finally:
        vector.close()


def test_vector_env_binding_rejects_oversized_or_nondivisible_batch() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    vector = DummyVecEnv([_env, _env])
    try:
        with pytest.raises(ValueError):
            PPOModel.create(
                vector,
                PPOConfig(n_steps=8, batch_size=32, n_epochs=1),
            )
        with pytest.raises(ValueError):
            PPOModel.create(
                vector,
                PPOConfig(n_steps=8, batch_size=6, n_epochs=1),
            )
    finally:
        vector.close()


def test_predict_still_rejects_batched_observation() -> None:
    env = _env()
    model = PPOModel.create(env, _config())
    observation, _ = env.reset()
    batched = np.stack([observation, observation])
    with pytest.raises(ValueError):
        model.predict(batched)


def test_load_validates_bound_vector_env_rollout_size(tmp_path) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    source_env = DummyVecEnv([_env, _env])
    target_env = DummyVecEnv([_env])
    try:
        model = PPOModel.create(
            source_env,
            PPOConfig(n_steps=6, batch_size=4, n_epochs=1),
        )
        archive = model.save(tmp_path / "vector-policy")
        loaded = PPOModel.load(archive, env=source_env, device="cpu")
        assert loaded._model.n_envs == 2
        with pytest.raises(ValueError):
            PPOModel.load(archive, env=target_env, device="cpu")

        inference_only = PPOModel.load(archive, env=None, device="cpu")
        assert inference_only._model.get_env() is None
    finally:
        source_env.close()
        target_env.close()


def test_learn_progress_hook_reports_actual_cumulative_timesteps() -> None:
    env = _env()
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"),
    )
    observed: list[int] = []
    try:
        assert model.learn(4) is model
        before = model.num_timesteps
        assert model.learn(4, progress_hook=observed.append) is model
        assert observed
        assert observed == sorted(observed)
        assert observed[0] > before
        assert observed[-1] == model.num_timesteps
    finally:
        env.close()


def test_checkpoint_can_be_saved_while_telemetry_observer_is_active(tmp_path, monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv

    import f1rl.learning.algorithms.ppo as ppo_module

    class Observer:
        def __init__(self):
            self.stream = (tmp_path / "exclusive-telemetry.jsonl").open("x", encoding="utf-8")

        def on_training_start(self, *args, **kwargs):
            pass

        def on_step(self, *args, **kwargs):
            pass

        def on_optimizer_step(self):
            pass

        def on_update(self, *args, **kwargs):
            pass

    monkeypatch.setattr(ppo_module, "_validate_training_admission", lambda _env: None)
    monkeypatch.setattr(ppo_module, "_local_demo_admitted", lambda _env: True)
    vector = DummyVecEnv([_env])
    model = PPOModel.create(vector, PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"))
    observer = Observer()
    archive = tmp_path / "checkpoint.zip"
    try:
        model.learn(4, telemetry_observer=observer,
                    progress_hook=lambda _step: model.save(archive) if not archive.exists() else None)
        assert archive.is_file()
    finally:
        observer.stream.close()
        vector.close()


def test_learn_none_progress_hook_preserves_existing_behavior() -> None:
    env = _env()
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"),
    )
    try:
        assert model.learn(4, progress_hook=None) is model
        assert model.num_timesteps >= 4
    finally:
        env.close()


def test_learn_progress_hook_exception_propagates_unchanged() -> None:
    class SentinelError(RuntimeError):
        pass

    env = _env()
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device="cpu"),
    )
    observed: list[int] = []

    def fail(step: int) -> None:
        observed.append(step)
        raise SentinelError("stop")

    try:
        with pytest.raises(SentinelError, match="stop"):
            model.learn(4, progress_hook=fail)
        assert observed
        assert observed[-1] == model.num_timesteps
    finally:
        env.close()


def test_learn_rejects_non_callable_progress_hook() -> None:
    model = PPOModel.create(_env(), PPOConfig(n_steps=4, batch_size=4, n_epochs=1))
    with pytest.raises(ValueError):
        model.learn(4, progress_hook=object())  # type: ignore[arg-type]


def test_telemetry_training_rejects_partial_ppo_rollout_before_learning(monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Observer:
        def on_training_start(self, *args, **kwargs):
            raise AssertionError('observer must not start for a partial rollout')
        def on_step(self, *args, **kwargs): pass
        def on_optimizer_step(self, *args, **kwargs): pass
        def on_update(self, *args, **kwargs): pass

    env = DummyVecEnv([_env])
    model = PPOModel.create(env, PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device='cpu'))
    monkeypatch.setattr(model._model, 'learn', lambda **kwargs: pytest.fail('training started'))
    try:
        with pytest.raises(ValueError, match='rollout'):
            model.learn(5, telemetry_observer=Observer())
    finally:
        env.close()


def test_telemetry_cleanup_preserves_learning_failure_and_attempts_both(monkeypatch):
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Observer:
        def on_training_start(self, *args, **kwargs): pass
        def on_step(self, *args, **kwargs): pass
        def on_optimizer_step(self, *args, **kwargs): pass
        def on_update(self, *args, **kwargs): pass

    env = DummyVecEnv([_env])
    model = PPOModel.create(env, PPOConfig(n_steps=4, batch_size=4, n_epochs=1, device='cpu'))
    calls = []
    def env_method(name, enabled):
        calls.append(enabled)
        if not enabled:
            raise OSError('telemetry disable failed')
    class Hook:
        def remove(self):
            raise OSError('hook removal failed')
    def learn(**kwargs):
        raise RuntimeError('primary learning failure')
    monkeypatch.setattr(env, 'env_method', env_method)
    monkeypatch.setattr(model._model, 'learn', learn)
    monkeypatch.setattr(model._model.policy.optimizer, 'register_step_post_hook', lambda f: Hook())
    try:
        with pytest.raises(RuntimeError, match='primary learning failure') as error:
            model.learn(4, telemetry_observer=Observer())
        assert calls == [True, False]
        assert 'hook removal failed' in ' '.join(error.value.__notes__)
        assert 'telemetry disable failed' in ' '.join(error.value.__notes__)
    finally:
        env.close()
