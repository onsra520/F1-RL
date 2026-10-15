"""Research-only TRAIN spawn preparation, not a scientific admission protocol.

The supplied registry is an explicit versioned input, never inferred from
which track happens to be loaded. Its hash is evidence of identity, not approval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import radians
from pathlib import PurePosixPath
import json
import random

from ..runtime.physics import PhysicsConfig
from ..sandbox.body_collision import sweep_body_against_obstacle
from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.geometry.vector import Vec3
from ..sandbox.multi_surface_support import certify_multisurface_support
from ..sandbox.track.model import TrackWorld
from ..sandbox.world import WorldState
from ..track_assets.canonical import fingerprint_track_world
from ..track_assets.model import TrackManifestEntry
from .corridor_acceptance import screen_authored_route_sweep
from .route_navigation import AuthoredRoute


class SpawnFeasibilityError(RuntimeError):
    """No registered feasible draw; never silently use the nominal spawn."""


@dataclass(frozen=True, slots=True)
class TrainResetConfig:
    seed: int
    training_registry: tuple[TrackManifestEntry, ...]
    max_attempts: int = 32
    version: str = "wp7_train_reset_research_v1"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32-1:
            raise ValueError("TRAIN perturbation seed must be a uint32")
        if self.version != "wp7_train_reset_research_v1":
            raise ValueError("unregistered TRAIN reset protocol")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 128:
            raise ValueError("feasible-spawn attempts must be within [1,128]")
        entries = self.training_registry
        if (not isinstance(entries, tuple) or not entries
                or any(not isinstance(e, TrackManifestEntry) for e in entries)
                or len({e.track_id for e in entries}) != len(entries)
                or len({e.world_sha256 for e in entries}) != len(entries)):
            raise ValueError("explicit unique TRAIN registry required")
        for entry in entries:
            path = PurePosixPath(entry.path)
            if (path.is_absolute() or "\\" in entry.path or ".." in path.parts
                    or len(path.parts) < 2 or path.parts[0] != "training"):
                raise ValueError("TRAIN reset registry cannot include HELD-OUT/non-training paths")

    def validate_track(self, track: TrackWorld, track_id: str) -> None:
        entry = next((e for e in self.training_registry if e.track_id == track_id), None)
        if entry is None or fingerprint_track_world(track) != entry.world_sha256:
            raise ValueError("TRAIN reset track identity is absent or mismatched")

    @property
    def sha256(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")).hexdigest()


def sample_train_spawn(
    config: TrainResetConfig, track_id: str, track: TrackWorld, initial: WorldState,
    physics: PhysicsConfig, route: AuthoredRoute, *, rank: int, reset_index: int,
) -> tuple[WorldState, dict[str, object]]:
    config.validate_track(track, track_id)
    if any(type(i) is not int or i < 0 for i in (rank, reset_index)):
        raise ValueError("rank/reset index must be nonnegative integers")
    if physics.mode != "wp7_suspension_v1" or not initial.vehicle.grounded:
        raise ValueError("TRAIN reset v1 requires grounded independent suspension mode")
    vehicle = initial.vehicle
    suspension = initial.suspension
    if (vehicle.velocity.magnitude > 1e-9 or vehicle.angular_velocity.magnitude > 1e-9
            or (vehicle.orientation.rotate(Vec3(0., 1., 0.))-Vec3(0., 1., 0.)).magnitude > 1e-9
            or any(abs(v) > 1e-9 for v in (
                suspension.heave_velocity_mps, suspension.pitch_x_rad, suspension.pitch_rate_radps,
                suspension.roll_z_rad, suspension.roll_rate_radps,
            ))):
        raise ValueError("TRAIN reset v1 requires an upright stationary nominal chassis")
    draw_key = f"{config.version}/{config.seed}/{rank}/{track_id}/{reset_index}"
    draw_seed = int.from_bytes(sha256(draw_key.encode("utf-8")).digest(), "big")
    rng = random.Random(draw_seed)
    right = vehicle.orientation.rotate(Vec3(1., 0., 0.))
    rejected = []
    for attempt in range(config.max_attempts):
        lateral, heading = rng.uniform(-.5, .5), rng.uniform(-5., 5.)
        candidate = replace(vehicle,
            position=vehicle.position + right*lateral,
            orientation=Quaternion.from_axis_angle(Vec3(0., 1., 0.), radians(heading))*vehicle.orientation)
        support = certify_multisurface_support(
            candidate, candidate, physics.body_geometry,
            physics.independent_suspension_policy.contact, track.surfaces,
        )
        reason = None
        if not support.valid:
            reason = f"road_support:{support.status}"
        elif any(sweep_body_against_obstacle(candidate, candidate, physics.body_geometry,
                                            o.volume).blocked for o in track.obstacles):
            reason = "body_obstacle"
        elif not screen_authored_route_sweep(route, vehicle.position, candidate.position).safe_for_route:
            reason = "route_uncertified"
        else:
            gates = (*track.checkpoints, *track.finish_lines)
            if any(g.volume.contains(candidate.position) for g in gates):
                reason = "spawn_inside_gate"
            next_gate = min(track.checkpoints, key=lambda c: c.order) if track.checkpoints else track.finish_lines[0]
            forward = next_gate.volume.orientation.rotate(Vec3(0., 0., 1.))
            if (candidate.position-next_gate.volume.center).dot(forward) >= -1e-9:
                reason = "spawn_not_before_next_gate"
        if reason is None:
            return replace(initial, vehicle=candidate), {
                "version": config.version, "configuration_sha256": config.sha256,
                "seed": config.seed, "rank": rank, "track_id": track_id,
                "reset_index": reset_index, "draw_seed": draw_seed,
                "world_sha256": fingerprint_track_world(track),
                "lateral_offset_m": lateral, "heading_jitter_deg": heading,
                "attempt": attempt, "rejected_draws": tuple(rejected),
                "scientific_evidence": False,
            }
        rejected.append({"lateral_offset_m": lateral, "heading_jitter_deg": heading, "reason": reason})
    raise SpawnFeasibilityError(f"no feasible TRAIN spawn in {config.max_attempts} draws: {rejected}")
