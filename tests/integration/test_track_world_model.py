from math import cos, pi, sin

from f1rl.sandbox.dynamics.integrator import integrate
from f1rl.sandbox.dynamics.vehicle_dynamics import compute_motion
from f1rl.sandbox.entities.boost_pad import BoostPad
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.score_pickup import ScorePickup
from f1rl.sandbox.entities.slow_zone import SlowZone
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState, collect_score_pickups
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 12.0, 20.0, 0.0, 0.02, 50.0, 9.81)


def track_with_jump() -> tuple[TrackWorld, PlanarSurfacePatch]:
    angle = pi / 6.0
    forward = Vec3(0.0, sin(angle), cos(angle))
    normal = Vec3(0.0, cos(angle), -sin(angle))
    ramp = PlanarSurfacePatch("ramp", Vec3(0.0, 1.0, 3.732050807568877), forward, normal, 4.0, 4.0)
    flat = PlanarSurfacePatch("flat", Vec3(0.0, 0.0, 0.0), Vec3(0, 0, 1), Vec3(0, 1, 0), 4.0, 4.0)
    landing = PlanarSurfacePatch("landing", Vec3(0.0, 0.0, 24.0), Vec3(0, 0, 1), Vec3(0, 1, 0), 6.0, 8.0)
    return TrackWorld(surfaces=(landing, ramp, flat)), ramp


def test_track_surface_query_drives_ramp_airtime_and_landing() -> None:
    world, ramp = track_with_jump()
    params = parameters()
    control = VehicleControl(0.0, 0.0, 0.0)
    ramp_point = ramp.origin + ramp.forward * 1.8
    state = VehicleState(
        ramp_point + ramp.normal * params.ride_height,
        Vec3(0.0, 0.0, 15.0),
        Quaternion.identity(),
        Vec3(0.0, 0.0, 0.0),
        True,
    )

    contact = world.surface_at(state.position, 0.5)
    assert contact is not None
    state = integrate(state, compute_motion(state, control, params, contact), params, contact, 0.1)
    assert state.velocity.y > 0.0

    gap_contact = world.surface_at(state.position, 0.5)
    assert gap_contact is None
    state = integrate(state, compute_motion(state, control, params, gap_contact), params, gap_contact, 0.05)
    assert state.grounded is False

    landed = False
    for _ in range(80):
        contact = world.surface_at(state.position, 0.8)
        state = integrate(state, compute_motion(state, control, params, contact), params, contact, 0.05)
        if contact is not None and state.grounded and state.position.z > 20.0:
            landed = True
            break

    assert landed
    assert state.velocity.y == 0.0


def interactive_world() -> TrackWorld:
    volume = OrientedBoxVolume(Vec3(0.0, 0.5, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity())
    return TrackWorld(
        obstacles=(Obstacle("obstacle", volume),),
        boost_pads=(BoostPad("boost", volume, 6.0),),
        slow_zones=(SlowZone("slow", volume, 2.0),),
        checkpoints=(Checkpoint("checkpoint", volume, 0),),
        finish_lines=(FinishLine("finish", volume),),
        score_pickups=(ScorePickup("black-card", volume, 100),),
    )


def test_world_interactions_and_score_pickup_consumption() -> None:
    world = interactive_world()
    initial = TrackState(frozenset())
    query = world.query_point(Vec3(0.0, 0.5, 0.0), 0.0, initial)

    assert tuple(item.id for item in query.obstacles) == ("obstacle",)
    assert tuple(item.id for item in query.boost_pads) == ("boost",)
    assert tuple(item.id for item in query.slow_zones) == ("slow",)
    assert tuple(item.id for item in query.checkpoints) == ("checkpoint",)
    assert tuple(item.id for item in query.finish_lines) == ("finish",)
    assert tuple(item.id for item in query.score_pickups) == ("black-card",)

    next_state, events = collect_score_pickups(query.score_pickups, initial)
    assert [(event.pickup_id, event.points) for event in events] == [("black-card", 100)]
    after = world.query_point(Vec3(0.0, 0.5, 0.0), 0.0, next_state)
    assert after.score_pickups == ()
    assert tuple(item.id for item in world.score_pickups) == ("black-card",)


def test_world_query_and_collection_are_exactly_deterministic() -> None:
    world = interactive_world()
    state = TrackState(frozenset())
    point = Vec3(0.0, 0.5, 0.0)

    first_query = world.query_point(point, 0.0, state)
    second_query = world.query_point(point, 0.0, state)
    assert first_query == second_query
    assert collect_score_pickups(first_query.score_pickups, state) == collect_score_pickups(
        second_query.score_pickups,
        state,
    )
