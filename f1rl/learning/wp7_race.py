"""Opt-in grounded numerical-trajectory race contract, not scientific approval.

Soft corridor is telemetry only. A hard violation invalidates the rollout;
no unapproved penalty, recovery law or successful termination is invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

from ..runtime.simulation import SimulationSnapshot
from ..sandbox.directed_center_crossing import directed_center_gate_crossing
from ..sandbox.center_trajectory import CenterSegment, linear_center_segments
from ..sandbox.interactions.events import CheckpointEntered, FinishLineEntered
from ..sandbox.track.model import TrackWorld
from .corridor_acceptance import screen_authored_route_sweep
from .episode import RaceProgressResult, RaceProgressState, RaceRoute, advance_race_progress
from .route_navigation import AuthoredRoute


class RaceContractError(RuntimeError):
    """Invalid research episode, never a successful/ordinary terminal sample."""


@dataclass(frozen=True, slots=True)
class WP7RaceConfig:
    version: str = "wp7_directed_grounded_v1"
    soft_corridor_m: float = 2.
    hard_corridor_m: float = 4.
    hard_violation: str = "reject_invalid_rollout"
    gate_orientation: str = "authored_local_positive_z_v1"

    def __post_init__(self) -> None:
        if (self.version not in ("wp7_directed_grounded_v1","wp7_directed_lifecycle_research_v1")
                or type(self.soft_corridor_m) not in (int, float) or self.soft_corridor_m != 2.
                or type(self.hard_corridor_m) not in (int, float) or self.hard_corridor_m != 4.
                or self.hard_violation != "reject_invalid_rollout"
                or self.gate_orientation != "authored_local_positive_z_v1"):
            raise ValueError("unregistered WP7 race contract or changed owner limits")

    def provenance(self, track: TrackWorld, route: AuthoredRoute) -> dict[str, object]:
        course = {
            "navigation": asdict(route.navigation),
            "checkpoints": [asdict(c) for c in sorted(track.checkpoints, key=lambda c: c.order)],
            "finish": asdict(track.finish_lines[0]),
        }
        document = {"configuration": asdict(self), "course": course}
        digest = sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()
        return {**document, "sha256": digest}

    def supports_physics_mode(self, mode: str) -> bool:
        if self.version == "wp7_directed_lifecycle_research_v1":
            return mode == "wp7_lifecycle_v1"
        return mode in ("wp7_suspension_v1","wp7_grounded_yaw_v1")


@dataclass(frozen=True, slots=True)
class DirectedRaceStep:
    progress: RaceProgressResult
    completed_checkpoint_ids: tuple[str, ...]
    directed_crossings: tuple[str, ...]
    max_sampled_deviation_m: float
    soft_zone_encountered: bool
    crossing_times_s: tuple[float, ...] = ()


def advance_directed_race(
    track: TrackWorld, route: RaceRoute, authored: AuthoredRoute,
    progress: RaceProgressState, snapshot: SimulationSnapshot, config: WP7RaceConfig | None = None,
) -> DirectedRaceStep:
    lifecycle = config is not None and config.version == "wp7_directed_lifecycle_research_v1"
    if lifecycle:
        audit = snapshot.physics_transition
        segments = snapshot.center_segments
        if (audit is None or audit.physical_mode != "wp7_lifecycle_v1"
                or not isinstance(segments,tuple) or not segments
                or not all(isinstance(s,CenterSegment) for s in segments)
                or abs(segments[0].start_time_s) > 1e-9
                or abs(segments[-1].end_time_s-(audit.end_time_s-audit.start_time_s)) > 1e-8
                or (segments[-1].end-snapshot.world.vehicle.position).magnitude > 1e-7
                or any(abs(a.end_time_s-b.start_time_s) > 1e-9
                       or (a.end-b.start).magnitude > 1e-7 for a,b in zip(segments,segments[1:]))):
            raise RaceContractError("lifecycle race requires a complete audited timed trajectory")
    else:
        suspension = snapshot.suspension_telemetry
        if suspension is None or snapshot.flight_telemetry is not None:
            raise RaceContractError("directed race requires certified grounded suspension substeps")
        poses = suspension.poses
        if any(not p.grounded for p in poses):
            raise RaceContractError("airborne race crossings are unregistered")
        segments = linear_center_segments(poses,0.,float(len(poses)-1))
    crossings = []
    peak = 0.
    soft = False
    gates = tuple((c.volume, CheckpointEntered(c.id, c.order), c.id) for c in track.checkpoints)
    gates += ((track.finish_lines[0].volume, FinishLineEntered(route.finish.id), route.finish.id),)
    for segment in segments:
        corridor = screen_authored_route_sweep(authored, segment.start, segment.end)
        if not corridor.safe_for_route:
            raise RaceContractError(f"WP7 route {corridor.status}; rollout invalid")
        peak = max(peak, corridor.max_sampled_deviation_m)
        soft |= corridor.soft_zone_encountered
        for volume, event, name in gates:
            if lifecycle:
                times = segment.directed_crossings(volume)
            else:
                crossing = directed_center_gate_crossing(segment.start,segment.end,volume)
                times = (segment.start_time_s+crossing.fraction,) if crossing is not None else ()
            for crossing_time in times:
                crossings.append((crossing_time, event, name))
    crossings.sort(key=lambda row: row[0])
    if any(abs(b[0]-a[0]) <= 1e-9 for a, b in zip(crossings, crossings[1:])):
        raise RaceContractError("simultaneous gate order is ambiguous; rollout invalid")
    current = progress
    completed = []
    for _, event, _ in crossings:
        result = advance_race_progress(route, current, (event,))
        current = result.state
        if result.completed_checkpoint_id is not None:
            completed.append(result.completed_checkpoint_id)
    result = RaceProgressResult(
        current, route.target_for(current), completed[-1] if completed else None,
        current.finished and not progress.finished,
    )
    return DirectedRaceStep(result, tuple(completed), tuple(c[2] for c in crossings), peak, soft,
                            tuple(c[0] for c in crossings) if lifecycle else ())
