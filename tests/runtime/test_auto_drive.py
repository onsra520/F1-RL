from itertools import pairwise
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError

from src.runtime.auto_drive import AutoDriveController
from src.runtime.drive_config import AutoDriveConfig
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleParameters
from src.tracks.model import AuthoredNavigation

VEHICLE = VehicleParameters(3.0, 0.5, 0.5, 6.0, 10.0, 0.4, 0.1, 15.0, 9.81)


def controller(points, position, *, heading=0.0, speed=0.0, config=None, obstacles=(), body_geometry=None, road_width=12.0):
    navigation = AuthoredNavigation(tuple(Vec3(*point) for point in points),
                                    tuple(f"road-{index}" for index in range(len(points) - 1)), ())
    env = SimpleNamespace(
        action_space=gym.spaces.Box(np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
                                    np.ones(3, dtype=np.float32)),
        runtime_snapshot=lambda: {
            "position": position,
            "heading_rad": heading,
            "speed_mps": speed,
            "next_checkpoint_index": 0,
        },
    )
    return AutoDriveController(env, navigation, VEHICLE, road_width, config or AutoDriveConfig(driving_style="centerline", target_max_speed_kmh=43.2), obstacles=obstacles, body_geometry=body_geometry)


def test_auto_drive_accelerates_and_keeps_straight_centerline():
    action = controller([(0, 0, 0), (0, 0, 50)], (0, 0.5, 2)).predict(None)

    assert action[0] == 0.0
    assert action[1] > 0.0
    assert action[2] == 0.0


def test_auto_drive_steers_into_turn_and_brakes_for_upcoming_curvature():
    action = controller([(0, 0, 0), (0, 0, 8), (8, 0, 8), (8, 0, 16)],
                        (0, 0.5, 7), speed=8).predict(None)

    assert action[0] > 0.0
    assert action[1] == 0.0
    assert action[2] > 0.0


def test_auto_drive_recovers_toward_route_at_reduced_speed():
    action = controller([(0, 0, 0), (0, 0, 50)], (3, 0.5, 10), speed=6).predict(None)

    assert action[0] < 0.0
    assert action[1] == 0.0
    assert action[2] > 0.0


def test_auto_drive_plans_speed_before_an_uphill_ramp():
    flat = controller([(0, 0, 0), (0, 0, 60)], (0, 0.5, 5))
    ramp = controller([(0, 0, 0), (0, 0, 15), (0, 3, 35), (0, 3, 60)], (0, 0.5, 5))

    assert flat._target_speed(5.0) == pytest.approx(12.0)
    assert ramp._target_speed(5.0) == 6.0


def test_default_config_belongs_to_right_lane_and_is_physics_capped():
    right = controller([(0, 0, 0), (0, 0, 50)], (0, 0.5, 2), config=AutoDriveConfig())
    center = controller([(0, 0, 0), (0, 0, 50)], (0, 0.5, 2),
                        config=AutoDriveConfig(driving_style="centerline"))
    assert right._target_point(15).x > 0
    assert center._target_point(15).x == 0
    assert right.predict(None)[0] > center.predict(None)[0]
    assert right.max_speed == pytest.approx(12.5)
    hyper = controller([(0, 0, 0), (0, 0, 50)], (0, 0.5, 2),
                       config=AutoDriveConfig(speed_profile="hyper", target_max_speed_kmh=240.0))
    assert hyper.max_speed == VEHICLE.max_ground_speed


def test_right_lane_stays_inside_narrow_road_with_vehicle_clearance():
    config = AutoDriveConfig(edge_margin_m=0.7, vehicle_width_m=1.5)
    p = controller([(0, 0, 0), (0, 0, 20)], (0, 0.5, 2), config=config)
    p.track_width = 2.0
    assert p._offset_at(5) == 0.0
    p.track_width = 4.0
    assert p._offset_at(5) <= 4 / 2 - config.vehicle_width_m / 2 - config.edge_margin_m


def test_right_lane_tangent_is_continuous_at_s_curve_vertex():
    config = AutoDriveConfig()
    p = controller([(0, 0, 0), (0, 0, 10), (4, 0, 20), (0, 0, 30), (0, 0, 40)],
                   (0, .5, 1), config=config)
    before = p._target_point(p.distances[2] - .01)
    after = p._target_point(p.distances[2] + .01)
    assert (after-before).magnitude < .1
    assert abs(p._offset_at(p.distances[2])) <= p.track_width / 2


def test_racing_and_right_have_distinct_corner_targets_and_speed_tuning():
    points = [(0, 0, 0), (0, 0, 12), (8, 0, 12), (8, 0, 20)]
    right = controller(points, (0, .5, 5), config=AutoDriveConfig())
    racing = controller(points, (0, .5, 5), config=AutoDriveConfig(driving_style="racing"))
    assert right._target_point(9) != racing._target_point(9)
    safe = controller(points, (0, .5, 5), config=AutoDriveConfig(corner_acceleration_mps2=1.0))
    faster = controller(points, (0, .5, 5), config=AutoDriveConfig(corner_acceleration_mps2=5.0))
    assert safe._target_speed(10) < faster._target_speed(10)


@pytest.mark.parametrize("updates", [
    {"target_max_speed_kmh": 0.0}, {"max_lookahead_m": 2.0, "min_lookahead_m": 8.0},
    {"edge_margin_m": -1.0}, {"speed_profile": "unlimited"},
    {"steep_grade_threshold": 0.08}, {"braking_factor": 1.1},
])
def test_invalid_drive_configuration_rejected(updates):
    with pytest.raises(ValidationError):
        AutoDriveConfig(**updates)

def test_racing_line_uses_opposite_corner_side_for_left_and_right_turns():
    config = AutoDriveConfig(driving_style="racing")
    left = controller([(0, 0, 0), (0, 0, 12), (-8, 0, 12), (-8, 0, 20)],
                      (0, .5, 5), config=config)
    right = controller([(0, 0, 0), (0, 0, 12), (8, 0, 12), (8, 0, 20)],
                       (0, .5, 5), config=config)
    assert left._offset_at(9) > 0
    assert right._offset_at(9) < 0

from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.entities.obstacle import Obstacle
from src.simulation.geometry.rotation import Quaternion
from src.simulation.track.volume import OrientedBoxVolume


def _physics_obstacle(x=2.8, y=0.66, z=50.0):
    return Obstacle("cone-real-hitbox", OrientedBoxVolume(
        Vec3(x, y, z), Vec3(0.65, 0.60, 0.65), Quaternion(1, 0, 0, 0),
    ))


def _body_hitbox():
    return VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0, 0.3, 0))


def test_racing_obstacle_planner_uses_physics_hitbox_and_anticipates_without_jump():
    cfg = AutoDriveConfig(driving_style="racing")
    route = [(0, 0, 0), (0, 0, 100)]
    bare = controller(route, (0, 0.5, 5), config=cfg)
    hazard = controller(route, (0, 0.5, 5), config=cfg,
                        obstacles=(_physics_obstacle(),), body_geometry=_body_hitbox())
    assert hazard._offset_at(5) == bare._offset_at(5) == 0.0
    assert hazard._offset_at(50) < 0.0
    assert hazard._offset_at(50) < bare._offset_at(50)
    assert hazard._target_speed(45) <= 5.0
    # Finite, smooth lane transition instead of steering/target teleport.
    values = [hazard._offset_at(i * 0.1) for i in range(900)]
    assert max(abs(b-a) for a, b in pairwise(values)) < 0.1
    assert hazard._offset_at(80) == 0.0


def test_racing_hitbox_rejects_obstacle_if_no_valid_road_clearance():
    cfg = AutoDriveConfig(driving_style="racing")
    with pytest.raises(ValueError, match="cannot certify a road-width-safe corridor"):
        controller([(0, 0, 0), (0, 0, 100)], (0, 0.5, 5), config=cfg,
                   obstacles=(_physics_obstacle(x=0.2),),
                   body_geometry=_body_hitbox(), road_width=4.0)


def test_different_elevation_obstacle_does_not_force_unneeded_swerve():
    cfg = AutoDriveConfig(driving_style="racing")
    a = controller([(0, 0, 0), (0, 0, 100)], (0, 0.5, 5), config=cfg,
                   obstacles=(_physics_obstacle(y=7.0),), body_geometry=_body_hitbox())
    assert a._hazards == ()
    assert a._offset_at(50) == 0.0


def test_right_hand_controller_does_not_change_with_racing_obstacle_planner():
    cfg = AutoDriveConfig()
    route = [(0, 0, 0), (0, 0, 100)]
    a = controller(route, (0, 0.5, 5), config=cfg)
    b = controller(route, (0, 0.5, 5), config=cfg,
                   obstacles=(_physics_obstacle(),),body_geometry=_body_hitbox())
    assert all(a._target_point(i) == b._target_point(i) for i in range(0, 100, 5))
