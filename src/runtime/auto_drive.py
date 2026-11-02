"""Development Auto Drive: bounded road-local trajectory tracking with physical speed caps."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import atan, atan2, cos, hypot, sin, sqrt

import numpy as np

from src.rl.route_navigation import AuthoredRoute
from src.runtime.drive_config import AutoDriveConfig
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.entities.obstacle import Obstacle
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleParameters
from src.tracks.model import AuthoredNavigation


@dataclass(frozen=True, slots=True)
class _ObstacleCorridor:
    progress_m: float
    safe_lateral_m: float
    obstacle_id: str




class AutoDriveController:
    def __init__(self, env, navigation: AuthoredNavigation, vehicle: VehicleParameters,
                 track_width: float, config: AutoDriveConfig | None = None, *,
                 obstacles: tuple[Obstacle, ...] = (),
                 body_geometry: VehicleBodyGeometry | None = None) -> None:
        self.env = env
        self.route = AuthoredRoute(navigation)
        self.vehicle = vehicle
        self.config = config or AutoDriveConfig()
        if not np.isfinite(track_width) or track_width <= 0.0:
            raise ValueError("Auto Drive requires a finite positive track width")
        self.track_width = track_width
        self.max_speed = min(vehicle.max_ground_speed, self.config.target_max_speed_kmh / 3.6)
        self.distances = [0.0]
        for first, second in zip(navigation.points, navigation.points[1:], strict=False):
            self.distances.append(self.distances[-1] + (second - first).magnitude)
        self.segment_grades = tuple(
            abs((second.y - first.y) / max(hypot(second.x - first.x, second.z - first.z), 1e-8))
            for first, second in zip(navigation.points, navigation.points[1:], strict=False)
        )
        self.corner_curvatures = self._corner_curvatures(navigation.points)
        self._hazards = self._obstacle_corridors(obstacles, body_geometry)

    def _corner_curvatures(self, points: tuple[Vec3, ...]) -> tuple[tuple[float, float, float], ...]:
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
            angle = atan2(cross, dot)
            span = (first_length + second_length) / 2.0
            curvature = 2.0 * sin(abs(angle) / 2.0) / span
            if curvature > 1e-5:
                corners.append((self.distances[index], curvature, -1.0 if cross < 0 else 1.0))
        return tuple(corners)

    def _obstacle_corridors(self, obstacles: tuple[Obstacle, ...],
                            body: VehicleBodyGeometry | None) -> tuple[_ObstacleCorridor, ...]:
        """Plan around the actual simulator OBBs, never the decorative meshes.

        A yaw allowance covers the turning body's longitudinal hull. This is
        a conservative *planner*, not collision certification: Physics SAT
        remains authoritative and unscored contacts remain invalid.
        """
        if self.config.driving_style != "racing" or not obstacles:
            return ()
        if body is None:
            raise ValueError("Racing obstacle avoidance requires registered body hitbox geometry")
        cfg = self.config
        road_room = max(0.0, self.track_width / 2.0
                        - max(cfg.vehicle_width_m / 2.0, body.width_m / 2.0)
                        - cfg.edge_margin_m)
        corridors = []
        for obstacle in obstacles:
            center = obstacle.volume.center
            projected = self.route.project(center, next_checkpoint_index=0)
            distance = projected.progress_meters
            point = self._point_at(distance)
            if abs(center.y - point.y) > obstacle.volume.half_extents.y + body.height_m + 0.5:
                continue  # Different deck/elevation: no lateral hazard.
            tx, tz = self._tangent_at(distance)
            nx, nz = tz, -tx
            lateral = (center.x - point.x) * nx + (center.z - point.z) * nz
            right = obstacle.volume.orientation.rotate(Vec3(1.0, 0.0, 0.0))
            forward = obstacle.volume.orientation.rotate(Vec3(0.0, 0.0, 1.0))
            projected_half_width = (
                obstacle.volume.half_extents.x * abs(right.x * nx + right.z * nz)
                + obstacle.volume.half_extents.z * abs(forward.x * nx + forward.z * nz)
            )
            # Include the registered body width, swept yaw envelope and a
            # separate planning safety margin, without changing either hitbox.
            envelope = (projected_half_width + body.width_m / 2.0
                        + min(0.65, body.length_m * 0.15) + 0.35)
            safe_lateral = (min(0.0, lateral - envelope - 0.35) if lateral >= 0.0
                            else max(0.0, lateral + envelope + 0.35))
            if abs(safe_lateral) > road_room:
                raise ValueError(
                    f"Auto Drive cannot certify a road-width-safe corridor around "
                    f"{obstacle.id}; move the obstacle in Build Editor"
                )
            corridors.append(_ObstacleCorridor(distance, safe_lateral, obstacle.id))
        return tuple(sorted(corridors, key=lambda row: row.progress_m))

    @staticmethod
    def _hazard_weight(distance: float, progress: float) -> float:
        """C1-smooth anticipation and recovery, never a 1-frame lane jump."""
        before = progress - distance
        if before >= 36.0 or before <= -23.0:
            return 0.0
        if before > 15.0:
            fraction = (36.0 - before) / 21.0
        elif before < -8.0:
            fraction = (before + 23.0) / 15.0
        else:
            return 1.0
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def _segment(self, distance: float) -> tuple[int, float]:
        distance = min(self.route.total_meters, max(0.0, distance))
        index = min(len(self.distances) - 2, max(0, bisect_right(self.distances, distance) - 1))
        length = max(self.distances[index + 1] - self.distances[index], 1e-8)
        return index, (distance - self.distances[index]) / length

    def _point_at(self, distance: float) -> Vec3:
        index, fraction = self._segment(distance)
        first, second = self.route.points[index:index + 2]
        return first + (second - first) * fraction

    def _tangent_at(self, distance: float) -> tuple[float, float]:
        # Smooth adjoining segment directions to avoid normal flips and hard turns at polyline vertices.
        span = min(1.0, max(0.25, self.track_width * 0.1))
        a = self._point_at(distance - span)
        b = self._point_at(distance + span)
        dx, dz = b.x - a.x, b.z - a.z
        norm = hypot(dx, dz)
        if norm < 1e-8:
            index, _ = self._segment(distance)
            first, second = self.route.points[index:index + 2]
            dx, dz = second.x - first.x, second.z - first.z
            norm = hypot(dx, dz)
        return dx / max(norm, 1e-8), dz / max(norm, 1e-8)

    def _offset_at(self, distance: float) -> float:
        cfg = self.config
        clearance = max(0.0, self.track_width / 2.0 - cfg.vehicle_width_m / 2.0 - cfg.edge_margin_m)
        if cfg.driving_style == "centerline" or clearance == 0.0:
            return 0.0
        if cfg.driving_style == "right":
            desired = cfg.lane_offset_m if cfg.lane_offset_m else min(self.track_width * 0.07, 0.8)
            return min(clearance, desired)
        # Inside of the next corner only, with bounded anticipation. No road-edge crossing.
        approaching = [(abs(cd - distance), curvature, sign)
                       for cd, curvature, sign in self.corner_curvatures
                       if -2.0 <= cd - distance <= max(10.0, cfg.max_lookahead_m * 2)]
        if not approaching:
            return self._avoid_obstacles(distance, 0.0)
        _, curvature, sign = min(approaching)
        racing_offset = (min(clearance * 0.75, self.track_width * 0.2,
                             1.0 / max(curvature, 1e-5) * 0.25) * sign)
        return self._avoid_obstacles(distance, racing_offset)

    def _avoid_obstacles(self, distance: float, racing_offset: float) -> float:
        if not self._hazards:
            return racing_offset
        nearest = max(
            ((self._hazard_weight(distance, h.progress_m), h)
             for h in self._hazards),
            key=lambda pair: pair[0],
        )
        weight, corridor = nearest
        if weight == 0.0:
            return racing_offset
        return racing_offset * (1.0 - weight) + corridor.safe_lateral_m * weight

    def _target_point(self, distance: float) -> Vec3:
        point = self._point_at(distance)
        tx, tz = self._tangent_at(distance)
        offset = self._offset_at(distance)
        # For heading +Z, road-local right is +X; negative cross denotes right turn.
        return Vec3(point.x + tz * offset, point.y, point.z - tx * offset)

    def _grade_at(self, distance: float) -> float:
        index, _ = self._segment(distance)
        first, second = self.route.points[index:index + 2]
        horizontal = hypot(second.x - first.x, second.z - first.z)
        return 0.0 if horizontal <= 1e-8 else (second.y - first.y) / horizontal

    def _target_speed(self, distance: float) -> float:
        cfg = self.config
        target = self.max_speed
        braking = self.vehicle.max_braking * cfg.braking_factor
        for corner_distance, curvature, _ in self.corner_curvatures:
            distance_to_corner = corner_distance - distance
            if distance_to_corner < -2.0:
                continue
            corner_speed = min(self.max_speed, sqrt(cfg.corner_acceleration_mps2 / curvature))
            braking_distance = max(0.0, distance_to_corner - cfg.corner_anticipation_m)
            target = min(target, sqrt(corner_speed * corner_speed + 2.0 * braking * braking_distance))
        first_grade = max(0, bisect_right(self.distances, distance) - 1)
        end_grade = min(len(self.segment_grades), bisect_right(self.distances, distance + cfg.grade_lookahead_m))
        grade = max(self.segment_grades[first_grade:end_grade] or (abs(self._grade_at(distance)),))
        if grade > cfg.steep_grade_threshold:
            target = min(target, cfg.steep_grade_speed_mps)
        elif grade > cfg.moderate_grade_threshold:
            target = min(target, cfg.moderate_grade_speed_mps)
        if self._hazards:
            for hazard in self._hazards:
                remaining = hazard.progress_m - distance
                if -8.0 <= remaining <= 36.0:
                    safe_speed = 5.0
                    if remaining > 18.0:
                        safe_speed += min(self.max_speed, 3.0) * (remaining - 18.0) / 18.0
                    target = min(target, safe_speed)
        return min(self.max_speed, max(0.5, target))

    def predict(self, _observation, *, deterministic: bool = True) -> np.ndarray:
        del deterministic
        snapshot = self.env.runtime_snapshot()
        position = Vec3(*snapshot["position"])
        projection = self.route.project(
            position, next_checkpoint_index=int(snapshot["next_checkpoint_index"] or 0),
        )
        cfg = self.config
        speed = float(snapshot["speed_mps"])
        lookahead = min(cfg.max_lookahead_m, max(cfg.min_lookahead_m,
                        cfg.min_lookahead_m + speed * cfg.speed_lookahead_s))
        if projection.off_path_meters > self.track_width * 0.25:
            lookahead = min(lookahead, cfg.min_lookahead_m + 2.0)
        target = self._target_point(projection.progress_meters + lookahead)
        heading = float(snapshot["heading_rad"])
        dx, dz = target.x - position.x, target.z - position.z
        lateral = dx * cos(heading) - dz * sin(heading)
        curvature = 2.0 * lateral / max(dx * dx + dz * dz, 1e-6)
        steering_angle = atan(self.vehicle.wheelbase * curvature)
        steering = float(np.clip(cfg.steering_gain * steering_angle /
                                 max(self.vehicle.max_steering_angle, 1e-6), -1.0, 1.0))

        target_speed = self._target_speed(projection.progress_meters)
        if projection.off_path_meters > self.track_width * 0.25:
            target_speed = min(target_speed, cfg.recovery_speed_mps)
        if projection.off_path_meters > self.track_width * 0.5:
            target_speed = min(target_speed, cfg.recovery_speed_mps * 0.5)
        desired_acceleration = float(np.clip((target_speed - speed) * 1.2,
                                            -self.vehicle.max_braking * cfg.braking_factor,
                                            self.vehicle.max_acceleration * cfg.acceleration_factor))
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
