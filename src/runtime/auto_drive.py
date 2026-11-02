from __future__ import annotations

from bisect import bisect_right
from math import atan, atan2, cos, hypot, sin, sqrt

import numpy as np

from src.rl.route_navigation import AuthoredRoute
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleParameters
from src.tracks.model import AuthoredNavigation


class AutoDriveController:
    GRADE_LOOKAHEAD_M = 20.0

    def __init__(self, env, navigation: AuthoredNavigation, vehicle: VehicleParameters, track_width: float) -> None:
        self.env = env
        self.route = AuthoredRoute(navigation)
        self.vehicle = vehicle
        self.distances = [0.0]
        for first, second in zip(navigation.points, navigation.points[1:], strict=False):
            self.distances.append(self.distances[-1] + (second - first).magnitude)
        self.segment_grades = tuple(
            abs((second.y - first.y) / max(hypot(second.x - first.x, second.z - first.z), 1e-8))
            for first, second in zip(navigation.points, navigation.points[1:], strict=False)
        )
        self.corner_curvatures = self._corner_curvatures(navigation.points)
        self.max_speed = min(vehicle.max_ground_speed, 12.0)
        if not np.isfinite(track_width) or track_width <= 0.0:
            raise ValueError("Auto Drive requires a finite positive track width")
        self.track_width = track_width

    def _corner_curvatures(self, points: tuple[Vec3, ...]) -> tuple[tuple[float, float], ...]:
        corners = []
        for index in range(1, len(points) - 1):
            before, point, after = points[index - 1:index + 2]
            first = (point.x - before.x, point.z - before.z)
            second = (after.x - point.x, after.z - point.z)
            first_length = hypot(*first)
            second_length = hypot(*second)
            if min(first_length, second_length) <= 1e-8:
                continue
            cross = first[0] * second[1] - first[1] * second[0]
            dot = first[0] * second[0] + first[1] * second[1]
            angle = abs(atan2(cross, dot))
            span = (first_length + second_length) / 2.0
            curvature = 2.0 * sin(angle / 2.0) / span
            if curvature > 1e-5:
                corners.append((self.distances[index], curvature))
        return tuple(corners)

    def _point_at(self, distance: float) -> Vec3:
        distance = min(self.route.total_meters, max(0.0, distance))
        index = min(len(self.distances) - 2, max(0, bisect_right(self.distances, distance) - 1))
        first, second = self.route.points[index:index + 2]
        span = self.distances[index + 1] - self.distances[index]
        return first + (second - first) * ((distance - self.distances[index]) / span)

    def _grade_at(self, distance: float) -> float:
        index = min(len(self.distances) - 2, max(0, bisect_right(self.distances, distance) - 1))
        first, second = self.route.points[index:index + 2]
        horizontal = hypot(second.x - first.x, second.z - first.z)
        return 0.0 if horizontal <= 1e-8 else (second.y - first.y) / horizontal

    def _target_speed(self, distance: float) -> float:
        target = self.max_speed
        braking = self.vehicle.max_braking * 0.65
        for corner_distance, curvature in self.corner_curvatures:
            distance_to_corner = corner_distance - distance
            if distance_to_corner < -2.0:
                continue
            corner_speed = min(self.max_speed, sqrt(2.2 / curvature))
            braking_distance = max(0.0, distance_to_corner - 1.5)
            target = min(target, sqrt(corner_speed * corner_speed + 2.0 * braking * braking_distance))
        first_grade = max(0, bisect_right(self.distances, distance) - 1)
        end_grade = min(len(self.segment_grades), bisect_right(self.distances, distance + self.GRADE_LOOKAHEAD_M))
        grade = max(self.segment_grades[first_grade:end_grade] or (abs(self._grade_at(distance)),))
        if grade > 0.2:
            target = min(target, 4.0)
        elif grade > 0.1:
            target = min(target, 6.0)
        return max(1.5, target)

    def predict(self, _observation, *, deterministic: bool = True) -> np.ndarray:
        del deterministic
        snapshot = self.env.runtime_snapshot()
        position = Vec3(*snapshot["position"])
        projection = self.route.project(
            position,
            next_checkpoint_index=int(snapshot["next_checkpoint_index"] or 0),
        )
        speed = float(snapshot["speed_mps"])
        lookahead = min(14.0, max(4.0, 4.0 + speed * 0.65))
        if projection.off_path_meters > self.track_width * 0.25:
            lookahead = min(lookahead, 6.0)
        target = self._point_at(projection.progress_meters + lookahead)
        heading = float(snapshot["heading_rad"])
        dx, dz = target.x - position.x, target.z - position.z
        lateral = dx * cos(heading) - dz * sin(heading)
        distance_squared = max(dx * dx + dz * dz, 1e-6)
        curvature = 2.0 * lateral / distance_squared
        steering_angle = atan(self.vehicle.wheelbase * curvature)
        steering = float(np.clip(steering_angle / self.vehicle.max_steering_angle, -1.0, 1.0))

        target_speed = self._target_speed(projection.progress_meters)
        if projection.off_path_meters > self.track_width * 0.25:
            target_speed = min(target_speed, 3.0)
        if projection.off_path_meters > self.track_width * 0.5:
            target_speed = min(target_speed, 1.5)
        desired_acceleration = float(np.clip((target_speed - speed) * 1.2,
                                            -self.vehicle.max_braking * 0.65,
                                            self.vehicle.max_acceleration * 0.65))
        grade = self._grade_at(projection.progress_meters)
        gravity_along_route = -self.vehicle.gravity * grade / sqrt(1.0 + grade * grade)
        required_acceleration = desired_acceleration - gravity_along_route
        throttle = float(np.clip(required_acceleration / max(self.vehicle.max_acceleration, 1e-6), 0.0, 1.0))
        brake = float(np.clip(-required_acceleration / max(self.vehicle.max_braking, 1e-6), 0.0, 1.0))
        if required_acceleration >= 0.0:
            brake = 0.0
        else:
            throttle = 0.0
        if len(self.env.action_space.shape) != 1 or self.env.action_space.shape[0] not in (2, 3):
            raise ValueError("Auto Drive requires the registered two or three value action contract")
        if self.env.action_space.shape[0] == 2:
            return np.asarray([steering, throttle], dtype=np.float32)
        return np.asarray([steering, throttle, brake], dtype=np.float32)
