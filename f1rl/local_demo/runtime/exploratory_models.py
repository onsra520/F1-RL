from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box


@dataclass(frozen=True, slots=True)
class ExploratoryModel:
    model_id: str
    sha256: str
    bytes: int


ARCHIVES = (
    ("wp7-calibration", "2026-10-11_205037_ppo-multitrack-seed0-sample0/ppo-multitrack-seed0-sample0-final-model.zip",
     "83afdbb0953f4eab8349737bdfab277074a908963ba65428e1f513ed149ad852"),
    ("wp7-confirmation", "2026-10-12_203835_ppo-multitrack-seed1-sample1/ppo-multitrack-seed1-sample1-final-model.zip",
     "c5daa4cb8dfde0f2a1eb94f0e83f6a2f8f7bdc47ecf7870e799c5f16640d64f6"),
)
EXPLORATORY_IDS = frozenset(item[0] for item in ARCHIVES)
EXPLORATORY_DIGESTS = {model_id: digest for model_id, _, digest in ARCHIVES}
SENSOR_LAYOUT = (
    ("left-30deg", -0.5, 0.8660254037844386),
    ("left-15deg", -0.25881904510252074, 0.9659258262890683),
    ("left-7p5deg", -0.13052619222005157, 0.9914448613738104),
    ("front", 0.0, 1.0),
    ("right-7p5deg", 0.13052619222005157, 0.9914448613738104),
    ("right-15deg", 0.25881904510252074, 0.9659258262890683),
    ("right-30deg", 0.5, 0.8660254037844386),
)
OBSERVATION_FEATURES = tuple(f"rangefinder:{name}" for name, _, _ in SENSOR_LAYOUT) + (
    "velocity_local_x", "velocity_local_y", "velocity_local_z", "target_direction_local_x",
    "target_direction_local_y", "target_direction_local_z", "target_distance", "target_available", "grounded",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_exploratory_models(root: Path) -> tuple[tuple[ExploratoryModel, ...], dict[str, Path], tuple[str, ...]]:
    models, paths, warnings = [], {}, []
    for parent in (root, *root.parents):
        archive_root = parent / ".archives" / "F1-RL" / "wp7-exploratory"
        if not archive_root.is_dir():
            continue
        try:
            resolved_root = archive_root.resolve(strict=True)
        except OSError:
            warnings.append("exploratory archive directory cannot be read")
            break
        if resolved_root != archive_root.absolute():
            warnings.append("exploratory archive directory must not redirect outside its expected location")
            break
        archive_root = resolved_root
        for model_id, relative, expected in ARCHIVES:
            candidate = archive_root / relative
            try:
                path = candidate.resolve(strict=True)
            except FileNotFoundError:
                warnings.append(f"{model_id}: archived checkpoint is unavailable")
                continue
            except OSError:
                warnings.append(f"{model_id}: checkpoint cannot be read")
                continue
            try:
                if not path.is_relative_to(archive_root) or not path.is_file() or file_sha256(path) != expected:
                    warnings.append(f"{model_id}: checkpoint integrity mismatch")
                    continue
                models.append(ExploratoryModel(model_id, expected, path.stat().st_size))
                paths[model_id] = path
            except OSError:
                warnings.append(f"{model_id}: checkpoint cannot be read")
        break
    else:
        warnings.extend(f"{model_id}: archived checkpoint is unavailable" for model_id, _, _ in ARCHIVES)
    return tuple(models), paths, tuple(warnings)


class BrakeZeroExploratoryPolicy:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, *, deterministic: bool = True):
        action = np.asarray(self.model.predict(observation, deterministic=deterministic), dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or not (-1 <= action[0] <= 1 and 0 <= action[1] <= 1):
            raise ValueError("exploratory policy returned an invalid steering/throttle action")
        return np.asarray((action[0], action[1], 0.0), dtype=np.float32)


def validate_observation_encoder(encoder) -> None:
    config = encoder.config
    sensors = config.sensor_suite.rangefinders
    if (encoder.feature_names != OBSERVATION_FEATURES or config.velocity_scale != 15.0
            or config.target_distance_scale != 80.0 or len(sensors) != len(SENSOR_LAYOUT)):
        raise ValueError("exploratory PPO observation semantics do not match training")
    for sensor, (name, x, z) in zip(sensors, SENSOR_LAYOUT, strict=True):
        direction = sensor.local_direction
        if (sensor.id != name or sensor.max_distance != 30.0 or not isclose(direction.x, x, rel_tol=0, abs_tol=1e-12)
                or not isclose(direction.y, 0.0, rel_tol=0, abs_tol=1e-12)
                or not isclose(direction.z, z, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("exploratory PPO sensor semantics do not match training")


def load_exploratory_policy(path: Path, expected_sha256: str, env, observation_encoder):
    if env is None:
        raise ValueError("exploratory PPO requires a development environment")
    validate_observation_encoder(observation_encoder)
    if file_sha256(path) != expected_sha256:
        raise ValueError("exploratory PPO checkpoint SHA-256 changed before loading")
    from f1rl.learning.algorithms.ppo import PPOModel

    observation_space, action_space = PPOModel.inspect_spaces(path)
    expected_action = Box(np.asarray([-1.0, 0.0], dtype=np.float32),
                          np.asarray([1.0, 1.0], dtype=np.float32), dtype=np.float32)
    expected_runtime_action = Box(np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
                                  np.asarray([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)
    if observation_space != env.observation_space:
        raise ValueError("exploratory PPO observation space does not match the runtime")
    if action_space != expected_action or env.action_space != expected_runtime_action:
        raise ValueError("exploratory PPO action contract does not match the brake-zero adapter")
    return BrakeZeroExploratoryPolicy(PPOModel.load(path, device="cpu"))
