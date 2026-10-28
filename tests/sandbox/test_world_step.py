from src.sandbox.entities.boost_pad import BoostPad
from src.sandbox.entities.checkpoint import Checkpoint
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.entities.score_pickup import ScorePickup
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from src.sandbox.interactions.state import InteractionState
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import ScorePickupCollected, TrackState
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.sandbox.world import WorldState, step_world


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 0.0, 20.0, 0.0, 0.0, 50.0, 9.81)


def road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch("road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), Vec3(0.0, 1.0, 0.0), 20.0, 100.0)


def volume(z: float, half_z: float = 0.25) -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(0.0, 0.3, z), Vec3(1.0, 1.0, half_z), Quaternion.identity())


def world_state(z: float, speed: float = 0.0) -> WorldState:
    return WorldState(
        VehicleState(Vec3(0.0, 0.3, z), Vec3(0.0, 0.0, speed), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True),
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def test_current_query_drives_modifiers_and_boost_motion() -> None:
    track = TrackWorld(surfaces=(road(),), boost_pads=(BoostPad("boost", volume(0.0, 1.0), 5.0),))
    result = step_world(track, world_state(0.0), VehicleControl(0.0, 0.0, 0.0), parameters(), 0.1, 0.5)
    assert result.modifiers.longitudinal_acceleration_bonus == 5.0
    assert result.state.vehicle.velocity.z == 0.5
    assert result.state.vehicle.position.z == 0.05


def test_zone_entered_during_movement_starts_effect_next_tick() -> None:
    track = TrackWorld(surfaces=(road(),), boost_pads=(BoostPad("boost", volume(0.0, 0.5), 5.0),))
    first = step_world(track, world_state(-2.0, 10.0), VehicleControl(0.0, 0.0, 0.0), parameters(), 0.15, 0.5)
    assert first.modifiers.longitudinal_acceleration_bonus == 0.0
    assert first.query.boost_pads[0].id == "boost"

    second = step_world(track, first.state, VehicleControl(0.0, 0.0, 0.0), parameters(), 0.1, 0.5)
    assert second.modifiers.longitudinal_acceleration_bonus == 5.0


def test_rejected_obstacle_candidate_cannot_trigger_or_collect() -> None:
    blocked = volume(1.0, 0.2)
    track = TrackWorld(
        surfaces=(road(),),
        obstacles=(Obstacle("obstacle", blocked),),
        checkpoints=(Checkpoint("checkpoint", blocked, 0),),
        finish_lines=(FinishLine("finish", blocked),),
        score_pickups=(ScorePickup("pickup", blocked, 100),),
    )
    initial = world_state(0.0, 10.0)
    result = step_world(track, initial, VehicleControl(0.0, 0.0, 0.0), parameters(), 0.1, 0.5)

    assert result.state.vehicle.position == initial.vehicle.position
    assert result.state.vehicle.velocity == Vec3(0.0, 0.0, 0.0)
    assert result.events == (ObstacleHit("obstacle"),)
    assert result.state.track == initial.track
    assert result.state.score == initial.score
    assert result.query.checkpoints == ()
    assert result.query.finish_lines == ()
    assert result.query.score_pickups == ()


def test_entry_pickup_score_and_event_order_use_accepted_query() -> None:
    shared = volume(0.0, 1.0)
    track = TrackWorld(
        surfaces=(road(),),
        checkpoints=(Checkpoint("b-check", shared, 2), Checkpoint("a-check", shared, 1)),
        finish_lines=(FinishLine("b-finish", shared), FinishLine("a-finish", shared)),
        score_pickups=(ScorePickup("b-pick", shared, 20), ScorePickup("a-pick", shared, 10)),
    )
    result = step_world(track, world_state(0.0), VehicleControl(0.0, 0.0, 0.0), parameters(), 0.1, 0.5)

    assert result.events == (
        CheckpointEntered("a-check", 1),
        CheckpointEntered("b-check", 2),
        FinishLineEntered("a-finish"),
        FinishLineEntered("b-finish"),
        ScorePickupCollected("a-pick", 10),
        ScorePickupCollected("b-pick", 20),
    )
    assert result.state.score == ScoreState(30)
    assert result.state.track.collected_pickup_ids == frozenset({"a-pick", "b-pick"})
    assert result.query.score_pickups == ()
    assert result.query.contact is not None
    assert result.query.contact.point.z == result.state.vehicle.position.z
