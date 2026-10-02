from dataclasses import replace
from pathlib import Path

from f1rl.learning.experiments import ExperimentDefinition
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState, step_world
from f1rl.track_assets import load_track_asset


TRACK_ROOT = Path("assets/tracks")
PROFILE = Path("configs/experiments/ppo-generalization-v1.yml")


def _wp1b_protocol() -> tuple[VehicleParameters, float, float]:
    definition = ExperimentDefinition.from_yaml(PROFILE)
    parameters = VehicleParameters(
        definition.vehicle.wheelbase,
        definition.vehicle.ride_height,
        definition.vehicle.max_steering_angle,
        definition.vehicle.max_acceleration,
        definition.vehicle.max_braking,
        definition.vehicle.rolling_drag,
        definition.vehicle.air_drag,
        definition.vehicle.max_ground_speed,
        definition.vehicle.gravity,
    )
    return (
        parameters,
        definition.simulation.fixed_dt,
        definition.simulation.max_surface_distance,
    )


def _state(z: float, speed: float) -> WorldState:
    return WorldState(
        VehicleState(
            Vec3(0.0, 0.5, z),
            Vec3(0.0, 0.0, speed),
            Quaternion.identity(),
            Vec3(0.0, 0.0, 0.0),
            True,
        ),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def test_canonical_slalom_boost_increases_post_step_speed() -> None:
    asset = load_track_asset(TRACK_ROOT / "training/slalom-v1.json")
    parameters, dt, max_surface_distance = _wp1b_protocol()
    initial = _state(36.0, 5.0)
    control = VehicleControl(0.0, 0.0, 0.0)

    boosted = step_world(
        asset.world,
        initial,
        control,
        parameters,
        dt,
        max_surface_distance,
    )
    baseline = step_world(
        replace(asset.world, boost_pads=()),
        initial,
        control,
        parameters,
        dt,
        max_surface_distance,
    )

    assert boosted.modifiers.longitudinal_acceleration_bonus == 3.0
    assert baseline.modifiers.longitudinal_acceleration_bonus == 0.0
    assert boosted.state.vehicle.speed > baseline.state.vehicle.speed


def test_canonical_right_turn_slow_zone_reduces_post_step_speed() -> None:
    asset = load_track_asset(TRACK_ROOT / "training/right-turn-v1.json")
    parameters, dt, max_surface_distance = _wp1b_protocol()
    initial = _state(50.0, 10.0)
    control = VehicleControl(0.0, 1.0, 0.0)

    slowed = step_world(
        asset.world,
        initial,
        control,
        parameters,
        dt,
        max_surface_distance,
    )
    baseline = step_world(
        replace(asset.world, slow_zones=()),
        initial,
        control,
        parameters,
        dt,
        max_surface_distance,
    )

    assert slowed.modifiers.rolling_drag_multiplier == 2.0
    assert baseline.modifiers.rolling_drag_multiplier == 1.0
    assert slowed.state.vehicle.speed < baseline.state.vehicle.speed
