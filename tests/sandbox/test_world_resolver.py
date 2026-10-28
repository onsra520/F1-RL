from src.sandbox.dynamics.modifiers import MotionModifiers
from src.sandbox.entities.boost_pad import BoostPad
from src.sandbox.entities.obstacle import Obstacle
from src.sandbox.entities.slow_zone import SlowZone
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.events import ObstacleHit
from src.sandbox.interactions.resolver import resolve_motion_modifiers, resolve_obstacle_block
from src.sandbox.surface import SurfaceContact
from src.sandbox.track.query import TrackQuery
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState


def volume() -> OrientedBoxVolume:
    return OrientedBoxVolume(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def query(*, contact=None, obstacles=(), boost_pads=(), slow_zones=()) -> TrackQuery:
    return TrackQuery(contact, tuple(obstacles), tuple(boost_pads), tuple(slow_zones), (), (), ())


def vehicle(position: Vec3, velocity: Vec3, grounded: bool, orientation=None) -> VehicleState:
    return VehicleState(position, velocity, orientation or Quaternion.identity(), Vec3(0.1, 0.2, 0.3), grounded)


def test_resolve_motion_modifiers_requires_supporting_contact() -> None:
    q = query(
        boost_pads=(BoostPad("boost", volume(), 5.0),),
        slow_zones=(SlowZone("slow", volume(), 2.0),),
    )
    assert resolve_motion_modifiers(q) == MotionModifiers()


def test_overlapping_boosts_sum_and_slows_multiply_independent_of_order() -> None:
    contact = SurfaceContact(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    boosts = (BoostPad("b", volume(), 4.0), BoostPad("a", volume(), 3.0))
    slows = (SlowZone("b", volume(), 2.0), SlowZone("a", volume(), 1.5))
    first = resolve_motion_modifiers(query(contact=contact, boost_pads=boosts, slow_zones=slows))
    second = resolve_motion_modifiers(query(contact=contact, boost_pads=tuple(reversed(boosts)), slow_zones=tuple(reversed(slows))))
    assert first == MotionModifiers(7.0, 3.0)
    assert second == first


def test_no_obstacle_accepts_candidate_unchanged() -> None:
    previous = vehicle(Vec3(0.0, 0.3, 0.0), Vec3(0.0, 0.0, 5.0), True)
    candidate = vehicle(Vec3(0.0, 0.3, 1.0), Vec3(0.0, 0.0, 6.0), True)
    accepted, events = resolve_obstacle_block(previous, candidate, query())
    assert accepted == candidate
    assert events == ()


def test_obstacle_hit_rolls_back_and_zeroes_motion_with_sorted_events() -> None:
    previous_orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 0.2)
    candidate_orientation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), 0.8)
    previous = vehicle(Vec3(1.0, 0.3, 2.0), Vec3(0.0, 0.0, 4.0), False, previous_orientation)
    candidate = vehicle(Vec3(2.0, 0.7, 3.0), Vec3(1.0, 2.0, 6.0), True, candidate_orientation)
    obstacles = (Obstacle("z", volume()), Obstacle("a", volume()))

    accepted, events = resolve_obstacle_block(previous, candidate, query(obstacles=obstacles))

    assert accepted.position == previous.position
    assert accepted.orientation == previous.orientation
    assert accepted.velocity == Vec3(0.0, 0.0, 0.0)
    assert accepted.angular_velocity == Vec3(0.0, 0.0, 0.0)
    assert accepted.grounded is previous.grounded
    assert events == (ObstacleHit("a"), ObstacleHit("z"))
