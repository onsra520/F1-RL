from dataclasses import FrozenInstanceError

import pytest

from src.simulation.geometry.vector import Vec3
from src.simulation.perception.rangefinder import RangefinderSpec
from src.simulation.perception.sensor_suite import SensorSuite
from src.rl.observation import NavigationTarget, ObservationConfig, ObservationEncoder


def _suite(*ids: str) -> SensorSuite:
    return SensorSuite(
        tuple(
            RangefinderSpec(sensor_id, Vec3(0.0, 0.0, 1.0), 10.0)
            for sensor_id in ids
        )
    )


def test_observation_config_accepts_valid_values_and_is_frozen() -> None:
    config = ObservationConfig(_suite("front"), 50.0, 100.0)
    assert config.velocity_scale == 50.0
    assert config.target_distance_scale == 100.0
    with pytest.raises(FrozenInstanceError):
        config.velocity_scale = 60.0  # type: ignore[misc]


def test_navigation_target_preserves_position_and_is_frozen() -> None:
    position = Vec3(1.0, 2.0, 3.0)
    target = NavigationTarget(position)
    assert target.position is position
    with pytest.raises(FrozenInstanceError):
        target.position = Vec3(0.0, 0.0, 0.0)  # type: ignore[misc]


@pytest.mark.parametrize("sensor_suite", [None, (), object()])
def test_observation_config_requires_sensor_suite(sensor_suite: object) -> None:
    with pytest.raises(ValueError):
        ObservationConfig(sensor_suite, 50.0, 100.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value",
    [
        ("velocity_scale", True),
        ("velocity_scale", "50"),
        ("velocity_scale", 0.0),
        ("velocity_scale", -1.0),
        ("velocity_scale", float("nan")),
        ("velocity_scale", float("inf")),
        ("target_distance_scale", True),
        ("target_distance_scale", "100"),
        ("target_distance_scale", 0.0),
        ("target_distance_scale", -1.0),
        ("target_distance_scale", float("nan")),
        ("target_distance_scale", float("inf")),
    ],
)
def test_observation_config_rejects_invalid_scales(field: str, value: object) -> None:
    kwargs = {"sensor_suite": _suite(), "velocity_scale": 50.0, "target_distance_scale": 100.0}
    kwargs[field] = value
    with pytest.raises(ValueError):
        ObservationConfig(**kwargs)  # type: ignore[arg-type]


_FIXED_NAMES = (
    "velocity_local_x",
    "velocity_local_y",
    "velocity_local_z",
    "target_direction_local_x",
    "target_direction_local_y",
    "target_direction_local_z",
    "target_distance",
    "target_available",
    "grounded",
)


def test_feature_names_preserve_sensor_declaration_order() -> None:
    encoder = ObservationEncoder(ObservationConfig(_suite("z-front", "a-left", "m-right"), 50.0, 100.0))
    assert encoder.feature_names == (
        "rangefinder:z-front",
        "rangefinder:a-left",
        "rangefinder:m-right",
        *_FIXED_NAMES,
    )
    assert encoder.size == len(encoder.feature_names) == 12


def test_empty_sensor_suite_still_has_nine_fixed_features() -> None:
    encoder = ObservationEncoder(ObservationConfig(_suite(), 50.0, 100.0))
    assert encoder.feature_names == _FIXED_NAMES
    assert encoder.size == len(_FIXED_NAMES) == 9
