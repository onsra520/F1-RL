"""Development Auto Drive: bounded road-local trajectory tracking with physical speed caps."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import atan, atan2, cos, hypot, sin, sqrt

import numpy as np

from f1rl.learning.route_navigation import AuthoredRoute
from f1rl.local_demo.runtime.drive_config import AutoDriveConfig
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.vehicle import VehicleParameters
from f1rl.track_assets.model import AuthoredNavigation

MAX_STEERING_CHANGE_PER_SECOND = 2.5
RIGHT_LANE_ROUTE_MARGIN_M = 0.5


@dataclass(frozen=True, slots=True)
class _ObstacleCorridor:
    progress_m: float
    safe_lateral_m: float
    obstacle_id: str
    half_length_m: float = 0.0




class AutoDriveController:
    def __init__(self, env, navigation: AuthoredNavigation, vehicle: VehicleParameters,
                 track_width: float, config: AutoDriveConfig | None = None, *,
                 obstacles: tuple[Obstacle, ...] = (),
                 checkpoints: tuple[Checkpoint, ...] = (),
                 body_geometry: VehicleBodyGeometry | None = None,
                 route_limit_m: float | None = None,
                 allow_compact_corridor: bool = False,
                 allow_centerline_avoidance: bool = False) -> None:
        self.env = env
        self.route = AuthoredRoute(navigation)
        self.vehicle = vehicle
        self.config = config or AutoDriveConfig()
        self._steering = 0.0
        self._steering_tick: int | None = None
        self._spawn_progress_m: float | None = None
        if not np.isfinite(track_width) or track_width <= 0.0:
            raise ValueError("Auto Drive requires a finite positive track width")
        self.track_width = track_width
        if route_limit_m is not None and (not np.isfinite(route_limit_m) or route_limit_m <= 0.3):
            raise ValueError("Auto Drive requires a valid route corridor")
        self.route_limit_m = route_limit_m
        self.allow_compact_corridor = allow_compact_corridor
        self.allow_centerline_avoidance = allow_centerline_avoidance
        self.vehicle_width = max(self.config.vehicle_width_m, body_geometry.width_m if body_geometry else 0.0)
        self.max_speed = min(vehicle.max_ground_speed, self.config.target_max_speed_kmh / 3.6)
        self.distances = [0.0]
        for first, second in zip(navigation.points, navigation.points[1:], strict=False):
            self.distances.append(self.distances[-1] + (second - first).magnitude)
        self.segment_grades = tuple(
            abs((second.y - first.y) / max(hypot(second.x - first.x, second.z - first.z), 1e-8))
            for first, second in zip(navigation.points, navigation.points[1:], strict=False)
        )
        self.corner_curvatures = self._corner_curvatures(navigation.points)
        self._checkpoint_limits = tuple(
            (distance, checkpoint.volume.half_extents.x)
            for distance, checkpoint in zip(self.route.checkpoint_meters,
                                             sorted(checkpoints, key=lambda item: item.order), strict=False)
        )
        self._narrow_checkpoints = tuple(
            distance for distance, half_width in self._checkpoint_limits
            if half_width < max(self.config.lane_offset_m or self.track_width * 0.25,
                                self.track_width * 0.2 if self.config.driving_style == "racing" else 0.0)
            + self.vehicle_width / 2.0 + self.config.edge_margin_m
        )
        self._hazards: tuple[_ObstacleCorridor, ...] = ()
        self._hazards = self._obstacle_corridors(obstacles, body_geometry)
        if self._hazards and any(abs(self._offset_at(point)) >= half_width
                                 for point, half_width in self._checkpoint_limits):
            raise ValueError("Auto Drive cannot find an obstacle-safe path through a checkpoint")
        self._speed_distances, self._speed_limits = self._build_speed_profile()

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

        The compact straight-road fallback uses a smaller clearance than the
        usual yaw envelope. Physics SAT remains authoritative; this planner
        does not certify collision-free motion.
        """
        if not obstacles or (self.config.driving_style == "centerline" and not self.allow_centerline_avoidance):
            return ()
        if body is None:
            raise ValueError("Obstacle avoidance requires registered body hitbox geometry")
        cfg = self.config
        road_room = max(0.0, self.track_width / 2.0
                        - max(cfg.vehicle_width_m / 2.0, body.width_m / 2.0)
                        - cfg.edge_margin_m)
        corridors = []
        projected_obstacles = []
        for obstacle in obstacles:
            center = obstacle.volume.center
            projected = self.route.project(center, next_checkpoint_index=len(self.route.checkpoint_meters))
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
            clearance = envelope + 0.350001
            if abs(lateral) - clearance > road_room:
                continue
            longitudinal = (obstacle.volume.half_extents.x * abs(right.x * tx + right.z * tz)
                            + obstacle.volume.half_extents.z * abs(forward.x * tx + forward.z * tz)
                            + body.length_m / 2.0 + 0.75)
            compact_clearance = projected_half_width + self.vehicle_width / 2.0 + 0.5
            projected_obstacles.append((distance, lateral, clearance, longitudinal, obstacle.id, compact_clearance))
        groups = []
        for obstacle in sorted(projected_obstacles, key=lambda row: row[0] - row[3]):
            start, end = obstacle[0] - obstacle[3], obstacle[0] + obstacle[3]
            if groups and start <= groups[-1][1]:
                groups[-1][0].append(obstacle)
                groups[-1][1] = max(groups[-1][1], end)
            else:
                groups.append([[obstacle], end])
        for group, end in groups:
            start = min(row[0] - row[3] for row in group)
            distance = (start + end) / 2.0
            preferred = self._offset_at(distance)
            allowed_room = min(road_room, self.route_limit_m - 0.05) if self.route_limit_m is not None else road_room
            candidates = []
            margins = [row[2] for row in group]
            for compact in (False, True):
                if compact:
                    if (not self.allow_compact_corridor or self.route_limit_m is None or len(group) != 1 or
                            any(start - 10.0 <= corner <= end + 10.0 for corner, _, _ in self.corner_curvatures)):
                        break
                    margins = [row[5] for row in group]
                candidates = [preferred, -allowed_room, allowed_room]
                candidates.extend(value for row, margin in zip(group, margins, strict=True)
                                  for value in (row[1] - margin, row[1] + margin))
                candidates.extend(value for point, half_width in self._checkpoint_limits if start <= point <= end
                                  for value in (max(0.0, half_width - 1e-6), -max(0.0, half_width - 1e-6)))
                candidates = [value for value in candidates if abs(value) <= allowed_room
                              and all(abs(value - row[1]) >= margin - 1e-9
                                      for row, margin in zip(group, margins, strict=True))
                              and all(abs(value) < half_width for point, half_width in self._checkpoint_limits
                                      if start <= point <= end)]
                if candidates:
                    break
            if not candidates:
                raise ValueError("Auto Drive cannot certify a road-width-safe corridor around "
                                 f"{', '.join(row[4] for row in group)}; move the obstacle in Build Editor")
            safe = min(candidates, key=lambda value: abs(value - preferred))
            corridors.append(_ObstacleCorridor(distance, safe, group[0][4], (end - start) / 2.0))
        return tuple(sorted(corridors, key=lambda row: row.progress_m))

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
        clearance = max(0.0, self.track_width / 2.0 - self.vehicle_width / 2.0 - cfg.edge_margin_m)
        if clearance == 0.0:
            return 0.0
        if cfg.driving_style == "centerline":
            return self._avoid_obstacles(distance, 0.0) if self.allow_centerline_avoidance else 0.0
        gate = max((max(0.0, min(1.0, (20.0 - abs(point - distance)) / 10.0))
                    for point in self._narrow_checkpoints), default=0.0)
        gate = gate * gate * (3.0 - 2.0 * gate)
        if cfg.driving_style == "right":
            desired = cfg.lane_offset_m if cfg.lane_offset_m else self.track_width * 0.25
            if not cfg.lane_offset_m:
                proximity = max((max(0.0, min(1.0, (40.0 - abs(corner - distance)) / 20.0))
                                 for corner, curvature, _ in self.corner_curvatures if curvature > 0.02),
                                default=0.0)
                proximity = proximity * proximity * (3.0 - 2.0 * proximity)
                desired -= (desired - min(desired, self.track_width * 0.10, 1.1)) * proximity
                entry = max(0.0, min(1.0, distance / 20.0))
                entry = entry * entry * (3.0 - 2.0 * entry)
                initial = min(desired, self.track_width * 0.10, 1.1)
                desired = initial + (desired - initial) * entry
            desired *= 1.0 - gate
            if self.route_limit_m is not None:
                desired = min(desired, max(0.0, self.route_limit_m - RIGHT_LANE_ROUTE_MARGIN_M))
            return self._avoid_obstacles(distance, min(clearance, desired))
        anticipation = max(10.0, cfg.max_lookahead_m * 2)
        weighted_offset = 0.0
        total_weight = 0.0
        for corner, curvature, sign in self.corner_curvatures:
            remaining = corner - distance
            entering = max(0.0, min(1.0, (anticipation - remaining) / 8.0))
            exiting = max(0.0, min(1.0, (remaining + 5.0) / 5.0))
            weight = entering * entering * (3.0 - 2.0 * entering)
            weight *= exiting * exiting * (3.0 - 2.0 * exiting)
            weighted_offset += weight * sign * min(clearance * 0.75, self.track_width * 0.2,
                                                   1.0 / max(curvature, 1e-5) * 0.25)
            total_weight += weight
        racing_offset = weighted_offset * (1.0 - gate) / max(1.0, total_weight)
        return self._avoid_obstacles(distance, racing_offset)

    def _avoid_obstacles(self, distance: float, racing_offset: float) -> float:
        if not self._hazards:
            return racing_offset
        first, last = self._hazards[0], self._hazards[-1]
        if distance < first.progress_m - first.half_length_m:
            fraction = (distance - (first.progress_m - first.half_length_m - 21.0)) / 21.0
            target = first.safe_lateral_m
        elif distance > last.progress_m + last.half_length_m:
            fraction = ((last.progress_m + last.half_length_m + 15.0) - distance) / 15.0
            target = last.safe_lateral_m
        else:
            for before, after in zip(self._hazards, self._hazards[1:], strict=False):
                end = before.progress_m + before.half_length_m
                start = after.progress_m - after.half_length_m
                if end < distance < start:
                    if start - end > 36.0:
                        if distance <= end + 15.0:
                            fraction = (end + 15.0 - distance) / 15.0
                            target = before.safe_lateral_m
                        elif distance >= start - 21.0:
                            fraction = (distance - (start - 21.0)) / 21.0
                            target = after.safe_lateral_m
                        else:
                            return racing_offset
                        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                        return racing_offset * (1.0 - fraction) + target * fraction
                    fraction = (distance - end) / (start - end)
                    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                    return before.safe_lateral_m * (1.0 - fraction) + after.safe_lateral_m * fraction
            return next(h.safe_lateral_m for h in self._hazards
                        if h.progress_m - h.half_length_m <= distance <= h.progress_m + h.half_length_m)
        fraction = max(0.0, min(1.0, fraction))
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        return racing_offset * (1.0 - fraction) + target * fraction

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

    def _path_curvature(self, distance: float) -> float:
        """Centered-chord road curvature with a conservative offset-radius cap.

        Sampling the changing *target* point directly overestimates curvature
        at Racing-Line lane changes, forcing a car to crawl on straight roads.
        Curvature is measured on the actual authored road instead, then
        inflated for the smaller inner turning radius of the lateral offset.
        Separate low-speed envelopes handle obstacle-driven lane changes.
        """
        window = 3.0
        previous = self._point_at(max(0.0, distance - window))
        point = self._point_at(distance)
        following = self._point_at(min(self.route.total_meters, distance + window))
        abx, abz = point.x - previous.x, point.z - previous.z
        bcx, bcz = following.x - point.x, following.z - point.z
        acx, acz = following.x - previous.x, following.z - previous.z
        a = hypot(abx, abz)
        b = hypot(bcx, bcz)
        c = hypot(acx, acz)
        if min(a, b, c) < 1e-6:
            return 0.0
        road_curvature = 2.0 * abs(abx * bcz - abz * bcx) / (a * b * c)
        offset = abs(self._offset_at(distance))
        return road_curvature / max(0.65, 1.0 - road_curvature * offset)

    def _local_speed_limit(self, distance: float) -> float:
        cfg = self.config
        curvature = self._path_curvature(distance)
        target = self.max_speed
        if curvature > 1e-6:
            target = min(target, sqrt(cfg.corner_acceleration_mps2 / curvature))

        # Preserve grade-dependent physical caps; no change to road or Physics.
        first = max(0, bisect_right(self.distances, distance) - 1)
        last = min(len(self.segment_grades), bisect_right(self.distances, distance + cfg.grade_lookahead_m))
        grade = max(self.segment_grades[first:last] or (abs(self._grade_at(distance)),))
        if grade > cfg.steep_grade_threshold:
            target = min(target, cfg.steep_grade_speed_mps)
        elif grade > cfg.moderate_grade_threshold:
            target = min(target, cfg.moderate_grade_speed_mps)

        # The hitbox-based obstacle corridor remains a hard planning cap.
        for hazard in self._hazards:
            remaining = hazard.progress_m - distance
            if -8.0 - hazard.half_length_m <= remaining <= 36.0 + hazard.half_length_m:
                safe_speed = 5.0
                if remaining > 18.0 + hazard.half_length_m:
                    safe_speed += min(self.max_speed, 3.0) * (remaining - 18.0 - hazard.half_length_m) / 18.0
                target = min(target, safe_speed)
        return max(0.5, target)

    def _build_speed_profile(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Braking-feasible backward speed envelope over the actual route.

        v_before² <= v_after² + 2*a_brake*distance. In contrast to a
        corner-vertex-only speed target, this lets the car accelerate on
        low-curvature sections while preparing for the next real bend.
        """
        length = self.route.total_meters
        steps = max(1, int(length / 2.0) + 1)
        distances = tuple(length * i / steps for i in range(steps + 1))
        limits = [self._local_speed_limit(s) for s in distances]
        braking = self.vehicle.max_braking * self.config.braking_factor
        for i in range(steps - 1, -1, -1):
            ds = distances[i + 1] - distances[i]
            feasible = sqrt(limits[i + 1] ** 2 + 2.0 * braking * ds)
            limits[i] = min(limits[i], feasible)
        return distances, tuple(limits)

    def _target_speed(self, distance: float) -> float:
        # Conservative spatial anticipation for controller response time,
        # rather than changing the underlying Physics braking strength.
        target_distance = min(self.route.total_meters,
                              max(0.0, distance + self.config.corner_anticipation_m))
        index = min(len(self._speed_distances) - 2,
                    max(0, bisect_right(self._speed_distances, target_distance) - 1))
        before = self._speed_distances[index]
        after = self._speed_distances[index + 1]
        fraction = (target_distance - before) / max(after - before, 1e-8)
        v0, v1 = self._speed_limits[index:index + 2]
        target = sqrt(max(0.0, v0 * v0 * (1 - fraction) + v1 * v1 * fraction))
        return min(self.max_speed, max(0.5, target))

    def requires_obstacle_assist(self, snapshot: dict) -> bool:
        if not self._hazards:
            return False
        projection = self.route.project(
            Vec3(*snapshot["position"]), next_checkpoint_index=int(snapshot["next_checkpoint_index"] or 0),
        )
        return any(-12.0 - hazard.half_length_m <= hazard.progress_m - projection.progress_meters
                   <= 35.0 + hazard.half_length_m for hazard in self._hazards)

    def predict(self, _observation, *, deterministic: bool = True) -> np.ndarray:
        del deterministic
        snapshot = self.env.runtime_snapshot()
        position = Vec3(*snapshot["position"])
        projection = self.route.project(
            position, next_checkpoint_index=int(snapshot["next_checkpoint_index"] or 0),
        )
        cfg = self.config
        speed = float(snapshot["speed_mps"])
        recovery_distance = projection.off_path_meters
        if cfg.driving_style == "right":
            lane_position = self._target_point(projection.progress_meters)
            recovery_distance = hypot(position.x - lane_position.x, position.z - lane_position.z)
            edge_clearance = max(0.0, self.track_width / 2.0 - self.vehicle_width / 2.0 - cfg.edge_margin_m)
            if projection.off_path_meters > edge_clearance:
                recovery_distance = max(recovery_distance, self.track_width * 0.25 + 1e-8)
        lookahead = min(cfg.max_lookahead_m, max(cfg.min_lookahead_m,
                        cfg.min_lookahead_m + speed * cfg.speed_lookahead_s))
        if recovery_distance > self.track_width * 0.25:
            lookahead = min(lookahead, cfg.min_lookahead_m + 2.0)
        target = self._target_point(projection.progress_meters + lookahead)
        heading = float(snapshot["heading_rad"])
        dx, dz = target.x - position.x, target.z - position.z
        lateral = dx * cos(heading) - dz * sin(heading)
        curvature = 2.0 * lateral / max(dx * dx + dz * dz, 1e-6)
        steering_angle = atan(self.vehicle.wheelbase * curvature)
        steering = float(np.clip(cfg.steering_gain * steering_angle /
                                 max(self.vehicle.max_steering_angle, 1e-6), -1.0, 1.0))
        tick = int(snapshot["tick"])
        if self._steering_tick is None or tick < self._steering_tick:
            self._steering = 0.0
            self._spawn_progress_m = projection.progress_meters
            elapsed_steps = 1
        else:
            elapsed_steps = tick - self._steering_tick
        steering_step = MAX_STEERING_CHANGE_PER_SECOND * float(snapshot["dt_s"]) * elapsed_steps
        steering = float(np.clip(steering, self._steering - steering_step, self._steering + steering_step))
        self._steering = steering
        self._steering_tick = tick

        target_speed = self._target_speed(projection.progress_meters)
        if self._spawn_progress_m is not None and any(
            hazard.progress_m + hazard.half_length_m >= self._spawn_progress_m
            and hazard.progress_m - self._spawn_progress_m <= 2.0 * hazard.half_length_m
            and projection.progress_meters <= hazard.progress_m + hazard.half_length_m + 2.0
            for hazard in self._hazards
        ):
            target_speed = min(target_speed, 1.0)
        if recovery_distance > self.track_width * 0.25:
            target_speed = min(target_speed, cfg.recovery_speed_mps)
        if recovery_distance > self.track_width * 0.5:
            target_speed = min(target_speed, cfg.recovery_speed_mps * 0.5)
        desired_acceleration = float(np.clip((target_speed - speed) * 1.2,
                                            -self.vehicle.max_braking * cfg.braking_factor,
                                            self.vehicle.max_acceleration * min(1.0, cfg.acceleration_factor * {"normal": 1.0, "racing": 1.25, "hyper": 1.5}[cfg.speed_profile])))
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
