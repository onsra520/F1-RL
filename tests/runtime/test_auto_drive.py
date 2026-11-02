from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from src.runtime.auto_drive import AutoDriveController
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleParameters
from src.tracks.model import AuthoredNavigation


VEHICLE = VehicleParameters(3.0, 0.5, 0.5, 6.0, 10.0, 0.4, 0.1, 15.0, 9.81)


def controller(points, position, *, heading=0.0, speed=0.0):
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
    return AutoDriveController(env, navigation, VEHICLE, 12.0)


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

    assert flat._target_speed(5.0) == 12.0
    assert ramp._target_speed(5.0) == 6.0
