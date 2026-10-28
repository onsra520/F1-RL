import ast
from pathlib import Path
import subprocess
import sys

import numpy as np

from stable_baselines3.common.env_checker import check_env

from src.rl import EpisodeConfig, ObservationConfig, ObservationEncoder, RewardComposer
from src.rl.algorithms.ppo import PPOConfig, PPOModel
from src.rl.gym_environment import F1RLEnv
from src.core.simulation import SimulationConfig
from src.simulation.entities.finish_line import FinishLine
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.perception.sensor_suite import SensorSuite
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.state import TrackState
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleParameters, VehicleState
from src.simulation.world import WorldState


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
    from src.rl.algorithms import PPOConfig as ExportedConfig
    from src.rl.algorithms import PPOModel as ExportedModel

    assert ExportedConfig is PPOConfig
    assert ExportedModel is PPOModel

    import src.rl.algorithms as algorithms
    assert algorithms.__all__ == ["PPOConfig", "PPOModel"]


def test_base_learning_package_does_not_reexport_optional_ppo() -> None:
    import src.rl as learning
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
import src.rl
assert not hasattr(src.rl, "PPOConfig")
assert not hasattr(src.rl, "PPOModel")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_model_framework_imports_are_limited_to_backends() -> None:
    sb3_owners: list[str] = []
    torch_owners: list[str] = []
    for path in Path("src").rglob("*.py"):
        modules = _direct_imports(path)
        if any(module == "stable_baselines3" or module.startswith("stable_baselines3.") for module in modules):
            sb3_owners.append(path.as_posix())
        if any(module == "torch" or module.startswith("torch.") for module in modules):
            torch_owners.append(path.as_posix())

    assert sorted(sb3_owners) == ["src/rl/algorithms/ppo.py", "src/training/vector_env.py"]
    assert torch_owners == []


def test_lower_layers_remain_free_of_model_framework_dependencies() -> None:
    paths = [
        *Path("src/simulation").rglob("*.py"),
        *Path("src/core").rglob("*.py"),
        Path("src/rl/gym_environment.py"),
        Path("src/rl/action.py"),
        Path("src/rl/observation.py"),
        *Path("src/rl/rewards").rglob("*.py"),
        *Path("src/rl/episode").rglob("*.py"),
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
    text = Path("src/rl/algorithms/README.md").read_text(encoding="utf-8").lower()
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
    text = Path("src/rl/README.md").read_text(encoding="utf-8").lower()
    assert "ppo baseline" in text
    assert "implemented" in text
    assert "vectorized" in text
    assert "evaluation" in text
    assert "generalization" in text
