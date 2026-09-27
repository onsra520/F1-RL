from f1rl.sandbox.entities.boost_pad import BoostPad
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.entities.slow_zone import SlowZone
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import ScorePickupCollected, TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState, step_world


def road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch("road", Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0), 20.0, 200.0)


def volume(z: float, half_z: float = 1.0) -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(0, 0.3, z), Vec3(1, 1, half_z), Quaternion.identity())


def parameters(*, rolling_drag: float = 0.0) -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 0.0, 20.0, rolling_drag, 0.0, 100.0, 9.81)

def state(z: float = 0.0, speed: float = 0.0, *, grounded: bool = True) -> WorldState:
    return WorldState(
        VehicleState(Vec3(0, 0.3, z), Vec3(0, 0, speed), Quaternion.identity(), Vec3(0, 0, 0), grounded),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def test_boost_and_delayed_zone_entry() -> None:
    boost = BoostPad("boost", volume(0.0, 1.0), 5.0)
    boosted = step_world(TrackWorld(surfaces=(road(),), boost_pads=(boost,)), state(), VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    baseline = step_world(TrackWorld(surfaces=(road(),)), state(), VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert boosted.state.vehicle.velocity.z > baseline.state.vehicle.velocity.z

    entered = step_world(TrackWorld(surfaces=(road(),), boost_pads=(boost,)), state(-2.0, 10.0), VehicleControl(0, 0, 0), parameters(), 0.15, 0.5)
    assert entered.modifiers.longitudinal_acceleration_bonus == 0.0
    assert entered.query.boost_pads
    next_step = step_world(TrackWorld(surfaces=(road(),), boost_pads=(boost,)), entered.state, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert next_step.modifiers.longitudinal_acceleration_bonus == 5.0


def test_slow_zone_and_airborne_ignore_track_modifiers() -> None:
    slow = SlowZone("slow", volume(0.0), 3.0)
    boosted = BoostPad("boost", volume(0.0), 10.0)
    params = parameters(rolling_drag=1.0)
    slowed = step_world(TrackWorld(surfaces=(road(),), slow_zones=(slow,)), state(0.0, 10.0), VehicleControl(0, 0, 0), params, 0.1, 0.5)
    baseline = step_world(TrackWorld(surfaces=(road(),)), state(0.0, 10.0), VehicleControl(0, 0, 0), params, 0.1, 0.5)
    assert slowed.state.vehicle.velocity.z < baseline.state.vehicle.velocity.z
    assert params.rolling_drag == 1.0

    air_track = TrackWorld(boost_pads=(boosted,), slow_zones=(slow,))
    air_state = state(0.0, 10.0, grounded=False)
    with_zones = step_world(air_track, air_state, VehicleControl(0, 0, 0), params, 0.1, 0.5)
    without_zones = step_world(TrackWorld(), air_state, VehicleControl(0, 0, 0), params, 0.1, 0.5)
    assert with_zones.state.vehicle == without_zones.state.vehicle

def test_checkpoint_and_finish_entry_leave_reentry() -> None:
    zone = volume(0.0, 0.5)
    track = TrackWorld(
        surfaces=(road(),),
        checkpoints=(Checkpoint("check", zone, 0),),
        finish_lines=(FinishLine("finish", zone),),
    )
    first = step_world(track, state(), VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert first.events == (CheckpointEntered("check", 0), FinishLineEntered("finish"))
    stay = step_world(track, first.state, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert stay.events == ()

    left_state = WorldState(
        VehicleState(Vec3(0, 0.3, 2.0), Vec3(0, 0, 0), Quaternion.identity(), Vec3(0, 0, 0), True),
        stay.state.track,
        stay.state.interactions,
        stay.state.score,
    )
    left = step_world(track, left_state, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert left.events == ()
    reentry_state = WorldState(state().vehicle, left.state.track, left.state.interactions, left.state.score)
    reentered = step_world(track, reentry_state, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert reentered.events == (CheckpointEntered("check", 0), FinishLineEntered("finish"))


def test_score_pickup_collects_and_scores_once() -> None:
    pickup = ScorePickup("card", volume(0.0), 100)
    track = TrackWorld(surfaces=(road(),), score_pickups=(pickup,))
    first = step_world(track, state(), VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert first.events == (ScorePickupCollected("card", 100),)
    assert first.state.score == ScoreState(100)
    assert first.query.score_pickups == ()

    second = step_world(track, first.state, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)
    assert second.events == ()
    assert second.state.score == ScoreState(100)

def test_obstacle_rejection_blocks_triggers_and_pickup() -> None:
    blocked = volume(1.0, 0.2)
    track = TrackWorld(
        surfaces=(road(),),
        obstacles=(Obstacle("b-obstacle", blocked), Obstacle("a-obstacle", blocked)),
        checkpoints=(Checkpoint("check", blocked, 0),),
        finish_lines=(FinishLine("finish", blocked),),
        score_pickups=(ScorePickup("card", blocked, 50),),
    )
    initial = state(0.0, 10.0)
    result = step_world(track, initial, VehicleControl(0, 0, 0), parameters(), 0.1, 0.5)

    assert result.state.vehicle.position == initial.vehicle.position
    assert result.state.vehicle.velocity == Vec3(0, 0, 0)
    assert result.events == (ObstacleHit("a-obstacle"), ObstacleHit("b-obstacle"))
    assert result.query.checkpoints == ()
    assert result.query.finish_lines == ()
    assert result.query.score_pickups == ()
    assert result.state.score == ScoreState()


def simulate_sequence() -> tuple:
    shared = volume(0.0, 2.0)
    track = TrackWorld(
        surfaces=(road(),),
        boost_pads=(BoostPad("boost", shared, 3.0),),
        slow_zones=(SlowZone("slow", shared, 1.5),),
        score_pickups=(ScorePickup("card", shared, 25),),
    )
    current = state(0.0, 5.0)
    results = []
    for _ in range(3):
        result = step_world(track, current, VehicleControl(0, 0, 0), parameters(rolling_drag=0.2), 0.1, 0.5)
        results.append(result)
        current = result.state
    return tuple(results)


def test_repeated_multi_step_simulation_is_exactly_deterministic() -> None:
    assert simulate_sequence() == simulate_sequence()
