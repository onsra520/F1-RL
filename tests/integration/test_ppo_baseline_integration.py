import ast
from pathlib import Path
import subprocess
import sys

import numpy as np

from stable_baselines3.common.env_checker import check_env

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


def _smoke_env() -> F1RLEnv:
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


def test_real_env_is_sb3_compatible_and_accepts_predicted_action() -> None:
    env = _smoke_env()
    check_env(env, warn=False, skip_render_check=True)
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=8, batch_size=4, n_epochs=1, device="cpu"),
    )
    observation, _ = env.reset(seed=0)

    action = model.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert env.action_space.contains(action)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["tick"] == 1
    assert np.isfinite(next_observation).all()


def test_short_real_learning_is_cumulative() -> None:
    env = _smoke_env()
    model = PPOModel.create(
        env,
        PPOConfig(n_steps=8, batch_size=4, n_epochs=1, seed=0, device="cpu"),
    )

    assert model.learn(8) is model
    first = model.num_timesteps
    assert first >= 8

    model.learn(8)
    second = model.num_timesteps
    assert second >= first + 8


def test_real_training_save_load_prediction_and_continue_round_trip(tmp_path) -> None:
    env = _smoke_env()
    config = PPOConfig(n_steps=8, batch_size=4, n_epochs=1, seed=0, device="cpu")
    model = PPOModel.create(env, config)
    model.learn(8)
    observation, _ = env.reset(seed=0)
    action_before = model.predict(observation, deterministic=True)

    archive = model.save(tmp_path / "ppo-smoke")
    fresh_env = _smoke_env()
    loaded = PPOModel.load(archive, env=fresh_env, device="cpu")
    action_after = loaded.predict(observation, deterministic=True)

    np.testing.assert_allclose(action_after, action_before, rtol=1e-6, atol=1e-7)
    fresh_observation, _ = fresh_env.reset(seed=0)
    assert np.allclose(fresh_observation, observation)
    next_observation, reward, terminated, truncated, _ = fresh_env.step(action_after)
    assert fresh_env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False

    before_continue = loaded.num_timesteps
    loaded.learn(8)
    assert loaded.num_timesteps >= before_continue + 8


def _direct_imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_algorithm_package_exports_only_approved_ppo_surface() -> None:
    from f1rl.learning.algorithms import PPOConfig as ExportedConfig
    from f1rl.learning.algorithms import PPOModel as ExportedModel

    assert ExportedConfig is PPOConfig
    assert ExportedModel is PPOModel

    import f1rl.learning.algorithms as algorithms
    assert algorithms.__all__ == ["PPOConfig", "PPOModel"]


def test_base_learning_package_does_not_reexport_optional_ppo() -> None:
    import f1rl.learning as learning
    assert not hasattr(learning, "PPOConfig")
    assert not hasattr(learning, "PPOModel")


def test_base_learning_import_does_not_require_sb3_or_torch() -> None:
    code = r'''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "stable_baselines3" or name.startswith("stable_baselines3."):
        raise ImportError("blocked stable_baselines3")
    if name == "torch" or name.startswith("torch."):
        raise ImportError("blocked torch")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import f1rl.learning
assert not hasattr(f1rl.learning, "PPOConfig")
assert not hasattr(f1rl.learning, "PPOModel")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_model_framework_imports_are_limited_to_backends_and_exploratory_guards() -> None:
    sb3_owners: list[str] = []
    torch_owners: list[str] = []
    for path in Path("f1rl").rglob("*.py"):
        modules = _direct_imports(path)
        if any(module == "stable_baselines3" or module.startswith("stable_baselines3.") for module in modules):
            sb3_owners.append(path.as_posix())
        if any(module == "torch" or module.startswith("torch.") for module in modules):
            torch_owners.append(path.as_posix())

    assert sorted(sb3_owners) == ["f1rl/exploratory/permit.py", "f1rl/exploratory/pilot.py",
        "f1rl/learning/algorithms/ppo.py", "f1rl/learning/training/vector_env.py"]
    assert sorted(torch_owners) == ["f1rl/exploratory/permit.py", "f1rl/exploratory/pilot.py"]


def test_lower_layers_remain_free_of_model_framework_dependencies() -> None:
    paths = [
        *Path("f1rl/sandbox").rglob("*.py"),
        *Path("f1rl/runtime").rglob("*.py"),
        Path("f1rl/learning/gym_environment.py"),
        Path("f1rl/learning/action.py"),
        Path("f1rl/learning/observation.py"),
        *Path("f1rl/learning/rewards").rglob("*.py"),
        *Path("f1rl/learning/episode").rglob("*.py"),
    ]
    forbidden = ("stable_baselines3", "torch")
    offenders: list[str] = []
    for path in paths:
        if any(module == prefix or module.startswith(prefix + ".") for module in _direct_imports(path) for prefix in forbidden):
            offenders.append(path.as_posix())
    assert offenders == []


def test_generated_policy_archive_is_ignored_and_untracked() -> None:
    env = _smoke_env()
    model = PPOModel.create(env, PPOConfig(n_steps=8, batch_size=4, n_epochs=1))
    archive = Path("artifacts/policies/pytest-ppo-baseline.zip")
    try:
        model.save(archive)
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(archive)],
            cwd=Path.cwd(),
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(archive)],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(archive)],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert ignored.returncode == 0
        assert status.stdout == ""
        assert tracked.returncode != 0
    finally:
        archive.unlink(missing_ok=True)


def test_algorithm_docs_describe_implemented_ppo_contract() -> None:
    text = Path("f1rl/learning/algorithms/README.md").read_text(encoding="utf-8").lower()
    for token in (
        "stable-baselines3",
        ".[rl]",
        "ppo-baseline.yml",
        "mlppolicy",
        "cuda",
        "auto",
        "hybrid",
        "save",
        "load",
        "vectorized",
    ):
        assert token in text


def test_learning_docs_mark_ppo_baseline_implemented_but_future_layers_pending() -> None:
    text = Path("f1rl/learning/README.md").read_text(encoding="utf-8").lower()
    assert "ppo baseline" in text
    assert "implemented" in text
    assert "vectorized" in text
    assert "evaluation" in text
    assert "generalization" in text
