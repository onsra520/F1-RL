from itertools import pairwise
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError

from f1rl.local_demo.runtime.auto_drive import AutoDriveController
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.vehicle import VehicleParameters
from f1rl.track_assets.model import AuthoredNavigation

VEHICLE = VehicleParameters(3.0, 0.5, 0.5, 6.0, 10.0, 0.4, 0.1, 15.0, 9.81)


def controller(points, position, *, heading=0.0, speed=0.0, config=None, obstacles=(), body_geometry=None, road_width=12.0, route_limit_m=None, allow_compact_corridor=False, allow_centerline_avoidance=False):
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
            "dt_s": 0.1,
            "tick": 0,
        },
    )
    return AutoDriveController(env, navigation, VEHICLE, road_width, config or AutoDriveConfig(driving_style="centerline", target_max_speed_kmh=43.2), obstacles=obstacles, body_geometry=body_geometry, route_limit_m=route_limit_m, allow_compact_corridor=allow_compact_corridor, allow_centerline_avoidance=allow_centerline_avoidance)


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


def test_steering_changes_are_bounded_between_physics_steps():
    drive = controller([(0, 0, 0), (0, 0, 50)], (0, .5, 5))
    before = drive.predict(None)
    drive.env.runtime_snapshot = lambda: {"position": (-4, .5, 5), "heading_rad": 0,
                                          "speed_mps": 0, "next_checkpoint_index": 0, "dt_s": 0.1, "tick": 1}
    after = drive.predict(None)
    assert abs(after[0] - before[0]) <= 0.250001
    assert drive.predict(None)[0] == after[0]
    drive.env.runtime_snapshot = lambda: {"position": (-4, .5, 5), "heading_rad": 0,
                                          "speed_mps": 0, "next_checkpoint_index": 0, "dt_s": 0.1, "tick": 0}
    assert drive.predict(None)[0] == after[0]


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


def test_racing_line_corner_target_enters_and_exits_continuously():
    drive = controller([(0, 0, 0), (0, 0, 60), (20, 0, 60)],
                       (0, .5, 5), config=AutoDriveConfig(driving_style="racing"))
    offsets = [drive._offset_at(i * 0.1) for i in range(701)]
    assert max(abs(b - a) for a, b in pairwise(offsets)) < 0.1
    assert min(offsets) < -1.0


def test_racing_line_blends_overlapping_opposite_corners():
    drive = controller([(0, 0, 0), (0, 0, 60), (10, 0, 60), (10, 0, 100)],
                       (0, .5, 5), config=AutoDriveConfig(driving_style="racing"))
    offsets = [drive._offset_at(i * 0.1) for i in range(901)]
    assert max(abs(b - a) for a, b in pairwise(offsets)) < 0.1
    assert min(offsets) < 0 < max(offsets)

from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.track.volume import OrientedBoxVolume


def _physics_obstacle(x=2.8, y=0.66, z=50.0):
    return Obstacle("cone-real-hitbox", OrientedBoxVolume(
        Vec3(x, y, z), Vec3(0.65, 0.60, 0.65), Quaternion(1, 0, 0, 0),
    ))


def _body_hitbox():
    return VehicleBodyGeometry(1.8, 1.2, 4.0, Vec3(0, 0.3, 0))


def test_centerline_obstacle_avoidance_is_opt_in_and_ignores_distant_obstacles():
    route = [(0, 0, 0), (0, 0, 100)]
    config = AutoDriveConfig(driving_style="centerline")
    standard = controller(route, (0, .5, 0), config=config, obstacles=(_physics_obstacle(x=0),), body_geometry=_body_hitbox())
    assisted = controller(route, (0, .5, 0), config=config, obstacles=(_physics_obstacle(x=0),), body_geometry=_body_hitbox(), allow_centerline_avoidance=True)
    distant = controller(route, (0, .5, 0), config=config, obstacles=(_physics_obstacle(x=15),), body_geometry=_body_hitbox(), allow_centerline_avoidance=True)
    assert standard._offset_at(50) == 0
    assert abs(assisted._offset_at(50)) > 0
    assert distant._offset_at(50) == 0
    assert not distant._hazards


def test_compact_corridor_respects_configured_vehicle_width():
    barrier = Obstacle("barrier", OrientedBoxVolume(
        Vec3(0, .93, 50), Vec3(2, .75, .35), Quaternion(1, 0, 0, 0),
    ))
    route = [(0, 0, 0), (0, 0, 100)]
    narrow = controller(route, (0, .5, 0), config=AutoDriveConfig(), obstacles=(barrier,),
                        body_geometry=_body_hitbox(), route_limit_m=4.0, allow_compact_corridor=True)
    assert abs(narrow._hazards[0].safe_lateral_m) == pytest.approx(3.4)
    with pytest.raises(ValueError, match="cannot certify"):
        controller(route, (0, .5, 0), config=AutoDriveConfig(), obstacles=(barrier,),
                   body_geometry=_body_hitbox(), route_limit_m=4.0)
    with pytest.raises(ValueError, match="cannot certify"):
        controller(route, (0, .5, 0), config=AutoDriveConfig(vehicle_width_m=3.2), obstacles=(barrier,),
                   body_geometry=_body_hitbox(), route_limit_m=4.0, allow_compact_corridor=True)


def test_obstacle_assistance_covers_both_ends_of_a_long_group():
    drive = controller([(0, 0, 0), (0, 0, 120)], (0, .5, 0), config=AutoDriveConfig())
    drive._hazards = (SimpleNamespace(progress_m=60.0, half_length_m=30.0),)
    for distance in (24.0, 35.0, 75.0, 100.0):
        assert drive.requires_obstacle_assist({"position": (0, .5, distance), "next_checkpoint_index": 0})
    assert not drive.requires_obstacle_assist({"position": (0, .5, 110.0), "next_checkpoint_index": 0})


def test_compact_corridor_refuses_overlapping_obstacles_and_nearby_turns():
    barrier = Obstacle("barrier", OrientedBoxVolume(
        Vec3(0, .93, 50), Vec3(2.4, .75, .35), Quaternion(1, 0, 0, 0),
    ))
    second = Obstacle("barrier-two", OrientedBoxVolume(
        Vec3(0, .93, 52), Vec3(2, .75, .35), Quaternion(1, 0, 0, 0),
    ))
    for points, obstacles in [([(0, 0, 0), (0, 0, 100)], (barrier, second)),
                              ([(0, 0, 0), (0, 0, 50), (50, 0, 50)], (barrier,))]:
        with pytest.raises(ValueError, match="cannot certify"):
            controller(points, (0, .5, 0), config=AutoDriveConfig(), obstacles=obstacles,
                       body_geometry=_body_hitbox(), route_limit_m=4.0, allow_compact_corridor=True)


def test_near_spawn_obstacle_group_slows_even_when_its_center_is_behind_spawn():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 10), speed=2,
                       config=AutoDriveConfig(), body_geometry=_body_hitbox(),
                       obstacles=tuple(_physics_obstacle(z=z) for z in (2, 8, 14)))
    action = drive.predict(None)
    assert action[1] == 0
    assert action[2] > 0


def test_obstacle_entirely_behind_spawn_does_not_trigger_near_spawn_slowdown():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 20), speed=2,
                       config=AutoDriveConfig(), body_geometry=_body_hitbox(),
                       obstacles=(_physics_obstacle(z=3),))
    action = drive.predict(None)
    assert action[1] > 0
    assert action[2] == 0


def test_near_spawn_slowdown_recalculates_after_reset():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 20), speed=2,
                       config=AutoDriveConfig(), body_geometry=_body_hitbox(),
                       obstacles=(_physics_obstacle(z=8.4),))
    assert drive.predict(None)[1] > 0
    drive.env.runtime_snapshot = lambda: {"position": (0, .5, 20), "heading_rad": 0,
                                          "speed_mps": 2, "next_checkpoint_index": 0, "dt_s": .1, "tick": 1}
    drive.predict(None)
    drive.env.runtime_snapshot = lambda: {"position": (0, .5, 2.6), "heading_rad": 0,
                                          "speed_mps": 2, "next_checkpoint_index": 0, "dt_s": .1, "tick": 0}
    assert drive.predict(None)[2] > 0


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


def test_right_hand_controller_avoids_obstacle_in_its_lane():
    cfg = AutoDriveConfig()
    route = [(0, 0, 0), (0, 0, 100)]
    a = controller(route, (0, 0.5, 5), config=cfg)
    b = controller(route, (0, 0.5, 5), config=cfg,
                   obstacles=(_physics_obstacle(),),body_geometry=_body_hitbox())
    assert b._target_point(5) == a._target_point(5)
    assert b._target_point(50).x < a._target_point(50).x - 2.0
    assert b._target_speed(45) <= 5.0
    assert b._target_point(80) == a._target_point(80)


def test_right_hand_recovers_when_near_road_edge():
    drive = controller([(0, 0, 0), (0, 0, 100)], (5.8, .5, 40), speed=6, config=AutoDriveConfig())
    action = drive.predict(None)
    assert action[1] == 0.0
    assert action[2] > 0.0


def test_right_lane_clearance_uses_physics_body_width():
    config = AutoDriveConfig(vehicle_width_m=0.5, edge_margin_m=0.2, lane_offset_m=5.0)
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 5), config=config,
                       road_width=4.0, body_geometry=_body_hitbox())
    assert drive._target_point(50).x == pytest.approx(0.9)


def test_right_hand_avoids_both_nearby_obstacles():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 5), config=AutoDriveConfig(),
                       obstacles=(_physics_obstacle(x=3), _physics_obstacle(x=-1)), body_geometry=_body_hitbox())
    lateral = drive._target_point(50).x
    assert abs(lateral - 3) >= 2.85
    assert abs(lateral + 1) >= 2.85


def test_right_hand_obstacle_targets_remain_continuous():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 5), config=AutoDriveConfig(),
                       obstacles=(_physics_obstacle(x=3, z=50), _physics_obstacle(x=4, z=60)),
                       body_geometry=_body_hitbox())
    assert abs(drive._offset_at(58.001) - drive._offset_at(58.0)) < 0.01


def test_racing_obstacle_targets_remain_continuous():
    drive = controller([(0, 0, 0), (0, 0, 100)], (0, .5, 5), config=AutoDriveConfig(driving_style="racing"),
                       obstacles=(_physics_obstacle(x=2.8, z=50), _physics_obstacle(x=0.5, z=60)),
                       body_geometry=_body_hitbox())
    assert abs(drive._offset_at(58.001) - drive._offset_at(58.0)) < 0.01


def test_long_obstacle_merges_all_overlapping_corridors():
    long_cone = Obstacle("long", OrientedBoxVolume(Vec3(-1, .66, 60), Vec3(.65, .6, 30), Quaternion(1, 0, 0, 0)))
    drive = controller([(0, 0, 0), (0, 0, 120)], (0, .5, 5), config=AutoDriveConfig(),
                       obstacles=(_physics_obstacle(x=3, z=30), _physics_obstacle(x=3, z=50), long_cone),
                       body_geometry=_body_hitbox())
    assert abs(drive._offset_at(33.401) - drive._offset_at(33.4)) < 0.01
    assert abs(drive._offset_at(30) + 1) >= 2.85


def test_speed_limit_covers_the_full_obstacle_cluster():
    obstacles = tuple(_physics_obstacle(x=3, z=10 + 6 * i) for i in range(15))
    drive = controller([(0, 0, 0), (0, 0, 120)], (0, .5, 5), config=AutoDriveConfig(),
                       obstacles=obstacles, body_geometry=_body_hitbox())
    assert drive._target_speed(10) <= 5.0
    assert drive._target_speed(90) <= 5.0


def test_speed_envelope_satisfies_lateral_acceleration_and_braking_distance():
    cfg = AutoDriveConfig(driving_style="right", target_max_speed_kmh=54.0,
                          corner_acceleration_mps2=2.2, braking_factor=0.65)
    ctrl = controller([(0, 0, 0), (0, 0, 65), (10, 0, 75),
                       (20, 0, 85), (20, 0, 155)], (0, 0.5, 5), config=cfg)
    s, v = ctrl._speed_distances, ctrl._speed_limits
    for i in range(len(s)):
        assert v[i] <= ctrl.max_speed + 1e-8
        assert v[i] ** 2 * ctrl._path_curvature(s[i]) <= cfg.corner_acceleration_mps2 + 1e-6
        if i + 1 < len(s):
            delta = s[i + 1] - s[i]
            assert v[i] ** 2 <= v[i + 1] ** 2 + (
                2 * ctrl.vehicle.max_braking * cfg.braking_factor * delta + 1e-6
            )
    assert ctrl._target_speed(20) > min(ctrl._target_speed(float(x)) for x in range(60, 90))


def test_profile_releases_speed_after_bend_and_obeys_physical_cap():
    cfg = AutoDriveConfig(driving_style="centerline", target_max_speed_kmh=240.0)
    ctrl = controller([(0, 0, 0), (0, 0, 65), (12, 0, 77),
                       (12, 0, 160)], (0, 0.5, 5), config=cfg)
    assert ctrl._target_speed(15) > ctrl._target_speed(65)
    assert ctrl._target_speed(125) > ctrl._target_speed(65)
    assert all(speed <= ctrl.vehicle.max_ground_speed for speed in ctrl._speed_limits)


def test_profile_respects_obstacle_speed_and_grade_limits_without_hitbox_change():
    cfg = AutoDriveConfig(driving_style="racing", target_max_speed_kmh=90.0)
    ctrl = controller([(0, 0, 0), (0, 0, 100)], (0, 0.5, 5), config=cfg,
                      obstacles=(_physics_obstacle(),), body_geometry=_body_hitbox())
    assert ctrl._target_speed(45) <= 5.0
    assert ctrl._hazards[0].obstacle_id == "cone-real-hitbox"
    assert ctrl._hazards[0].safe_lateral_m < 0
    flat = controller([(0, 0, 0), (0, 0, 80)], (0, 0.5, 5))
    assert flat._target_speed(5) >= ctrl._target_speed(45)


def test_speed_presets_change_actual_throttle_without_changing_physics_or_vehicle_cap():
    """At equal speed target, the driver profiles must produce measured distinct control."""
    actions = {}
    for name in ("normal", "racing", "hyper"):
        c = controller([(0, 0, 0), (0, 0, 150)], (0, 0.5, 2),
                       config=AutoDriveConfig(driving_style="centerline",
                                              speed_profile=name,
                                              target_max_speed_kmh=120.0))
        actions[name] = c.predict(None)
        assert c.max_speed == VEHICLE.max_ground_speed
    assert 0 < actions["normal"][1] < actions["racing"][1] < actions["hyper"][1] <= 1.0
    assert all(action[2] == 0 for action in actions.values())


def test_default_right_lane_targets_lane_center_without_triggering_recovery():
    drive = controller([(0, 0, 0), (0, 0, 80)], (3, 0.5, 20), speed=7, config=AutoDriveConfig())

    assert drive._target_point(30).x == pytest.approx(3.0)
    action = drive.predict(None)
    assert action[1] > 0.0
    assert action[2] == 0.0


def test_right_lane_reduces_offset_smoothly_near_a_turn():
    drive = controller([(0, 0, 0), (0, 0, 100), (30, 0, 100), (100, 0, 100)],
                       (0, 0.5, 2), config=AutoDriveConfig())
    offsets = [drive._offset_at(i * 0.5) for i in range(280)]

    assert offsets[0] == pytest.approx(1.1)
    assert offsets[100] == pytest.approx(3.0)
    assert offsets[200] <= 1.1
    assert max(abs(b - a) for a, b in pairwise(offsets)) < 0.15


def test_default_right_lane_leaves_margin_inside_hard_route_corridor():
    drive = controller(
        [(0, 0, 0), (0, 0, 80)], (0, 0.5, 20), road_width=16.0,
        route_limit_m=4.0, config=AutoDriveConfig(driving_style="right"),
    )

    assert drive._offset_at(30.0) == pytest.approx(3.5)


@pytest.mark.parametrize("lane_offset", [0.0, 2.5])
def test_right_lane_targets_center_of_a_narrow_checkpoint(lane_offset):
    base = controller([(0, 0, 0), (0, 0, 50), (0, 0, 100)], (0, .5, 5), config=AutoDriveConfig())
    navigation = AuthoredNavigation(base.route.points, ("road-0", "road-1"), (1,))
    gate = Checkpoint("gate", OrientedBoxVolume(Vec3(0, 0, 50), Vec3(2, 2, 1), Quaternion(1, 0, 0, 0)), 0)
    drive = AutoDriveController(base.env, navigation, VEHICLE, 12, AutoDriveConfig(lane_offset_m=lane_offset), checkpoints=(gate,))

    assert drive._target_point(50).x == pytest.approx(0.0)


def test_saved_lane_offset_still_respects_checkpoint_aperture():
    base = controller([(0, 0, 0), (0, 0, 50), (0, 0, 100)], (0, .5, 5))
    navigation = AuthoredNavigation(base.route.points, ("road-0", "road-1"), (1,))
    gate = Checkpoint("gate", OrientedBoxVolume(Vec3(0, 0, 50), Vec3(4.5, 2, 1), Quaternion(1, 0, 0, 0)), 0)
    drive = AutoDriveController(base.env, navigation, VEHICLE, 12, AutoDriveConfig(lane_offset_m=5.0), checkpoints=(gate,))
    assert abs(drive._target_point(50).x) < 4.5


@pytest.mark.parametrize("lane_offset", [0.0, 0.1])
def test_racing_line_targets_center_of_a_narrow_checkpoint_before_corner(lane_offset):
    base = controller([(0, 0, 0), (0, 0, 45), (0, 0, 60), (20, 0, 60)], (0, .5, 5))
    navigation = AuthoredNavigation(base.route.points, ("road-0", "road-1", "road-2"), (1,))
    gate = Checkpoint("gate", OrientedBoxVolume(Vec3(0, 0, 45), Vec3(2, 2, 1), Quaternion(1, 0, 0, 0)), 0)
    drive = AutoDriveController(base.env, navigation, VEHICLE, 12, AutoDriveConfig(driving_style="racing", lane_offset_m=lane_offset), checkpoints=(gate,))
    assert drive._target_point(45).x == pytest.approx(0.0)


def test_checkpoint_obstacle_conflict_fails_closed():
    base = controller([(0, 0, 0), (0, 0, 50), (0, 0, 100)], (0, .5, 5))
    navigation = AuthoredNavigation(base.route.points, ("road-0", "road-1"), (1,))
    gate = Checkpoint("gate", OrientedBoxVolume(Vec3(0, 0, 50), Vec3(2, 2, 1), Quaternion(1, 0, 0, 0)), 0)
    with pytest.raises(ValueError, match="road-width-safe corridor"):
        AutoDriveController(base.env, navigation, VEHICLE, 12, AutoDriveConfig(), checkpoints=(gate,),
                            obstacles=(_physics_obstacle(x=0),), body_geometry=_body_hitbox())


def test_obstacle_transition_cannot_miss_a_nearby_checkpoint():
    base = controller([(0, 0, 0), (0, 0, 55), (0, 0, 100)], (0, .5, 5))
    navigation = AuthoredNavigation(base.route.points, ("road-0", "road-1"), (1,))
    gate = Checkpoint("gate", OrientedBoxVolume(Vec3(0, 0, 55), Vec3(2, 2, 1), Quaternion(1, 0, 0, 0)), 0)
    with pytest.raises(ValueError, match="checkpoint"):
        AutoDriveController(base.env, navigation, VEHICLE, 12, AutoDriveConfig(), checkpoints=(gate,),
                            obstacles=(_physics_obstacle(x=0),), body_geometry=_body_hitbox())
