import inspect
from pathlib import Path

from tests.dependency_rules import imports_project_layers

import src.core.simulation as runtime_simulation
from src.core.simulation import Simulation, SimulationConfig
from src.simulation.entities.checkpoint import Checkpoint
from src.simulation.entities.finish_line import FinishLine
from src.simulation.entities.score_pickup import ScorePickup
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.interactions.state import InteractionState
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.simulation.world import WorldState


def _road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch(
        "road",
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0),
        10.0,
        100.0,
    )


def _initial_world() -> WorldState:
    return WorldState(
        VehicleState(
            Vec3(0.0, 0.3, 0.0),
            Vec3(0.0, 0.0, 0.0),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def _parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)


def _interactive_track() -> TrackWorld:
    shared = OrientedBoxVolume(
        Vec3(0.0, 0.3, 0.0),
        Vec3(1.0, 1.0, 1.0),
        Quaternion.identity(),
    )
    return TrackWorld(
        surfaces=(_road(),),
        checkpoints=(Checkpoint("checkpoint", shared, 0),),
        finish_lines=(FinishLine("finish", shared),),
        score_pickups=(ScorePickup("pickup", shared, 50),),
    )


def _simulation(track: TrackWorld | None = None) -> Simulation:
    return Simulation(
        track or TrackWorld(surfaces=(_road(),)),
        _initial_world(),
        _parameters(),
        SimulationConfig(0.1, 0.5),
    )


def test_reset_restores_exact_initial_world_after_interactions() -> None:
    track = _interactive_track()
    initial = _initial_world()
    simulation = Simulation(track, initial, _parameters(), SimulationConfig(0.1, 0.5))

    changed = simulation.step(VehicleControl(0.0, 0.0, 0.0))
    assert changed.world is not initial
    assert changed.world.track.collected_pickup_ids == frozenset({"pickup"})
    assert changed.world.interactions.active_checkpoint_ids == frozenset({"checkpoint"})
    assert changed.world.interactions.active_finish_line_ids == frozenset({"finish"})
    assert changed.world.score == ScoreState(50)

    reset = simulation.reset()
    assert reset.tick == 0
    assert reset.elapsed_time == 0.0
    assert reset.world is initial
    assert reset.events == ()
    assert reset.world.track == TrackState(frozenset())
    assert reset.world.interactions == InteractionState(frozenset(), frozenset())
    assert reset.world.score == ScoreState()
    assert tuple(item.id for item in reset.query.score_pickups) == ("pickup",)


def _controls() -> tuple[VehicleControl, ...]:
    return (
        VehicleControl(0.0, 1.0, 0.0),
        VehicleControl(0.2, 0.8, 0.0),
        VehicleControl(-0.1, 0.6, 0.0),
        VehicleControl(0.0, 0.0, 0.2),
        VehicleControl(0.0, 0.5, 0.0),
    )


def test_identical_control_sequences_produce_identical_snapshots() -> None:
    left = _simulation()
    right = _simulation()

    left_snapshots = tuple(left.step(control) for control in _controls())
    right_snapshots = tuple(right.step(control) for control in _controls())

    assert left_snapshots == right_snapshots


def test_caller_interleaving_does_not_change_simulation_result() -> None:
    sequential_a = _simulation()
    sequential_b = _simulation()
    interleaved_a = _simulation()
    interleaved_b = _simulation()

    for control in _controls():
        sequential_a.step(control)
    for control in _controls():
        sequential_b.step(control)
    for control in _controls():
        interleaved_a.step(control)
        interleaved_b.step(control)

    assert sequential_a.snapshot == interleaved_a.snapshot
    assert sequential_b.snapshot == interleaved_b.snapshot


def test_runtime_public_surface_has_no_start_stop_or_scheduler_dependencies() -> None:
    simulation = _simulation()
    assert hasattr(simulation, "step")
    assert hasattr(simulation, "reset")
    assert hasattr(simulation, "snapshot")
    assert not hasattr(simulation, "start")
    assert not hasattr(simulation, "stop")

    assert not imports_project_layers(Path(inspect.getsourcefile(runtime_simulation)), ("rl", "training", "evaluation"))
    source = inspect.getsource(runtime_simulation).lower()
    forbidden = (
        "src.rl",
        "src.training",
        "src.evaluation",
        "src.api",
        "src.drivers",
        "fastapi",
        "websocket",
        "time.sleep",
        "threading",
        "asyncio",
    )
    assert all(token not in source for token in forbidden)
