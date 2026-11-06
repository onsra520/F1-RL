import pytest

import f1rl.runtime.simulation as runtime_simulation
from f1rl.runtime.simulation import Simulation, SimulationConfig
from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState, WorldStepResult, step_world


def _road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        100.0,
    )


def _world(speed: float = 0.0) -> WorldState:
    vehicle = VehicleState(
        Vec3(0.0, 0.3, 0.0),
        Vec3(0.0, 0.0, speed),
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


def _simulation(track: TrackWorld | None = None, world: WorldState | None = None) -> Simulation:
    return Simulation(
        track or TrackWorld(surfaces=(_road(),)),
        world or _world(),
        _parameters(),
        SimulationConfig(0.1, 0.5),
    )


def test_one_step_advances_exactly_one_tick_and_fixed_time() -> None:
    simulation = _simulation()
    result = simulation.step(VehicleControl(0.0, 1.0, 0.0))
    assert result.tick == 1
    assert result.elapsed_time == 0.1
    assert result is simulation.snapshot


def test_two_steps_use_tick_times_fixed_dt() -> None:
    simulation = _simulation()
    control = VehicleControl(0.0, 1.0, 0.0)

    first = simulation.step(control)
    second = simulation.step(control)

    assert (first.tick, first.elapsed_time) == (1, 0.1)
    assert (second.tick, second.elapsed_time) == (2, 0.2)


def test_step_preserves_sandbox_world_query_and_events() -> None:
    pickup = ScorePickup(
        "pickup",
        OrientedBoxVolume(Vec3(0.0, 0.3, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity()),
        25,
    )
    track = TrackWorld(surfaces=(_road(),), score_pickups=(pickup,))
    initial = _world()
    parameters = _parameters()
    config = SimulationConfig(0.1, 0.5)
    control = VehicleControl(0.0, 0.0, 0.0)
    expected = step_world(track, initial, control, parameters, config.fixed_dt, config.max_surface_distance)
    simulation = Simulation(track, initial, parameters, config)

    actual = simulation.step(control)
    assert actual.world == expected.state
    assert actual.query == expected.query
    assert actual.events == expected.events
    assert actual.events


def test_step_calls_step_world_once_with_configured_values(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation = _simulation()
    calls: list[tuple[float, float]] = []

    def fake_step_world(track, state, control, parameters, dt, max_surface_distance):
        calls.append((dt, max_surface_distance))
        return WorldStepResult(state, simulation.snapshot.query, (), ())

    monkeypatch.setattr(runtime_simulation, "step_world", fake_step_world, raising=False)
    simulation.step(VehicleControl(0.0, 0.0, 0.0))

    assert calls == [(0.1, 0.5)]


def test_elapsed_time_is_derived_from_tick_multiplication(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation = _simulation()

    def fake_step_world(track, state, control, parameters, dt, max_surface_distance):
        return WorldStepResult(state, simulation.snapshot.query, (), ())

    monkeypatch.setattr(runtime_simulation, "step_world", fake_step_world, raising=False)
    for _ in range(10):
        simulation.step(VehicleControl(0.0, 0.0, 0.0))

    assert simulation.snapshot.tick == 10
    assert simulation.snapshot.elapsed_time == 10 * 0.1


def test_failed_step_preserves_snapshot_identity_after_prior_success(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation = _simulation()
    control = VehicleControl(0.0, 0.0, 0.0)
    simulation.step(control)
    before = simulation.snapshot

    class SentinelError(RuntimeError):
        pass

    def fail_step_world(*args, **kwargs):
        raise SentinelError("boom")

    monkeypatch.setattr(runtime_simulation, "step_world", fail_step_world, raising=False)

    with pytest.raises(SentinelError, match="boom"):
        simulation.step(control)

    assert simulation.snapshot is before
    assert simulation.snapshot == before
