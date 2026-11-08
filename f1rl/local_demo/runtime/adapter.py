from __future__ import annotations

from math import atan2

from f1rl.sandbox.geometry.vector import Vec3


class RuntimeEnvironmentAdapter:
    def __init__(self, environment) -> None:
        self._environment = environment

    def __getattr__(self, name: str):
        return getattr(self._environment, name)

    def reset(self, **kwargs):
        return self._environment.reset(**kwargs)

    def step(self, action):
        return self._environment.step(action)

    def close(self) -> None:
        self._environment.close()

    def runtime_snapshot(self) -> dict[str, object]:
        environment = self._environment
        progress = environment._progress
        if progress is None:
            raise RuntimeError("reset() must be called before requesting a runtime snapshot")

        snapshot = environment._simulation.snapshot
        vehicle = snapshot.world.vehicle
        forward = vehicle.orientation.rotate(Vec3(0.0, 0.0, 1.0))
        route_progress = 0.0
        route_progress_fraction = 0.0
        if environment._authored_route is not None:
            route_progress = environment._authored_route.project(
                vehicle.position,
                next_checkpoint_index=progress.next_checkpoint_index,
            ).progress_meters
            route_progress_fraction = min(
                1.0, route_progress / environment._authored_route.total_meters
            )

        return {
            "schema_version": "f1rl_runtime_snapshot_v1",
            "tick": snapshot.tick,
            "time_s": snapshot.elapsed_time,
            "dt_s": environment._simulation_config.fixed_dt,
            "next_checkpoint_index": progress.next_checkpoint_index,
            "position": (float(vehicle.position.x), float(vehicle.position.y), float(vehicle.position.z)),
            "velocity": (float(vehicle.velocity.x), float(vehicle.velocity.y), float(vehicle.velocity.z)),
            "orientation_wxyz": (
                float(vehicle.orientation.w), float(vehicle.orientation.x),
                float(vehicle.orientation.y), float(vehicle.orientation.z),
            ),
            "grounded": bool(vehicle.grounded),
            "speed_mps": float(vehicle.speed),
            "heading_rad": atan2(forward.x, forward.z),
            "yaw_rate_radps": float(vehicle.angular_velocity.y),
            "route_progress_m": float(route_progress),
            "route_progress_fraction": float(route_progress_fraction),
            "route_highwater_m": float(environment._route_highwater),
            "checkpoint_count": len(environment._route.checkpoints),
            "physical_event_types": tuple(type(event).__name__ for event in snapshot.events),
        }
