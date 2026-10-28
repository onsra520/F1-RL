from dataclasses import FrozenInstanceError

import pytest

from src.runtime.simulation import SimulationConfig, SimulationSnapshot, SimulationState
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.state import InteractionState
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.query import TrackQuery
from src.sandbox.track.state import TrackState
from src.sandbox.vehicle import VehicleState
from src.sandbox.world import WorldState


def _world() -> WorldState:
    vehicle = VehicleState(
        position=Vec3(0.0, 0.3, 0.0),
        velocity=Vec3(0.0, 0.0, 0.0),
        orientation=Quaternion.identity(),
        angular_velocity=Vec3(0.0, 0.0, 0.0),
        grounded=True,
    )
    return WorldState(
        vehicle=vehicle,
        track=TrackState(frozenset()),
        interactions=InteractionState(frozenset(), frozenset()),
        score=ScoreState(),
    )


def _query() -> TrackQuery:
    return TrackQuery(None, (), (), (), (), (), ())


def test_simulation_config_is_validated_and_immutable() -> None:
    config = SimulationConfig(0.1, 0.5)
    assert config.fixed_dt == 0.1
    assert config.max_surface_distance == 0.5
    with pytest.raises(FrozenInstanceError):
        config.fixed_dt = 0.2  # type: ignore[misc]


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf"), True])
def test_simulation_config_rejects_invalid_fixed_dt(value: object) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(value, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_simulation_config_rejects_invalid_surface_distance(value: object) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(0.1, value)  # type: ignore[arg-type]


def test_simulation_config_accepts_zero_surface_distance() -> None:
    assert SimulationConfig(0.1, 0.0).max_surface_distance == 0.0


@pytest.mark.parametrize("tick", [-1, 1.5, True])
def test_simulation_state_rejects_invalid_tick(tick: object) -> None:
    with pytest.raises(ValueError):
        SimulationState(tick, 0.0, _world())  # type: ignore[arg-type]


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf"), True])
def test_simulation_state_rejects_invalid_elapsed_time(elapsed: object) -> None:
    with pytest.raises(ValueError):
        SimulationState(0, elapsed, _world())  # type: ignore[arg-type]


def test_simulation_snapshot_is_immutable_and_preserves_values() -> None:
    world = _world()
    query = _query()
    snapshot = SimulationSnapshot(0, 0.0, world, query, ())
    assert snapshot.world is world
    assert snapshot.query is query
    assert snapshot.events == ()
    with pytest.raises(FrozenInstanceError):
        snapshot.tick = 1  # type: ignore[misc]
