import pytest

from src.core.simulation import Simulation, SimulationConfig
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.vehicle import VehicleParameters, VehicleState
from src.simulation.world import WorldState


def _track() -> TrackWorld:
    road = PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        20.0,
    )
    return TrackWorld(surfaces=(road,))


def _world() -> WorldState:
    vehicle = VehicleState(
        Vec3(0.0, 0.3, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Quaternion.identity(),
        Vec3(0.0, 0.0, 0.0),
        True,
    )
    return WorldState(
        vehicle,
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def _parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)


def test_constructor_builds_tick_zero_snapshot_from_initial_world_query() -> None:
    track = _track()
    initial = _world()
    config = SimulationConfig(0.1, 0.5)
    simulation = Simulation(track, initial, _parameters(), config)

    expected_query = track.query_point(initial.vehicle.position, 0.5, initial.track)
    assert simulation.snapshot.tick == 0
    assert simulation.snapshot.elapsed_time == 0.0
    assert simulation.snapshot.world is initial
    assert simulation.snapshot.query == expected_query
    assert simulation.snapshot.query.contact is not None
    assert simulation.snapshot.events == ()


def test_snapshot_property_is_read_only() -> None:
    simulation = Simulation(_track(), _world(), _parameters(), SimulationConfig(0.1, 0.5))
    with pytest.raises(AttributeError):
        simulation.snapshot = simulation.snapshot  # type: ignore[misc]


def test_reset_at_tick_zero_restores_exact_initial_world_and_fresh_query() -> None:
    track = _track()
    initial = _world()
    simulation = Simulation(track, initial, _parameters(), SimulationConfig(0.1, 0.5))
    before = simulation.snapshot

    reset = simulation.reset()

    assert reset is simulation.snapshot
    assert reset is not before
    assert reset.tick == 0
    assert reset.elapsed_time == 0.0
    assert reset.world is initial
    assert reset.query == track.query_point(initial.vehicle.position, 0.5, initial.track)
    assert reset.events == ()


def test_construction_and_reset_do_not_mutate_domain_inputs() -> None:
    track = _track()
    initial = _world()
    parameters = _parameters()
    track_before = repr(track)
    world_before = repr(initial)
    parameters_before = repr(parameters)

    simulation = Simulation(track, initial, parameters, SimulationConfig(0.1, 0.5))
    simulation.reset()

    assert repr(track) == track_before
    assert repr(initial) == world_before
    assert repr(parameters) == parameters_before
