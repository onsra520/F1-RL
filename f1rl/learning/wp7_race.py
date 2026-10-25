"""Opt-in grounded numerical-trajectory race contract, not scientific approval.

Soft corridor is telemetry only. Historical v1 hard violations invalidate
the rollout. Owner-v2 versions terminate as valid failure and score only the
eligible event prefix, without adding a terminal reward adjustment.
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
from .hard_route_outcome import first_hard_route_exit, owner_outcome_contract
from .crash_terminal_registered import RACE_V4, descriptor
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
        if (self.version not in ("wp7_directed_grounded_v1","wp7_directed_lifecycle_research_v1",
                                "wp7_directed_grounded_owner_v2","wp7_directed_lifecycle_owner_v2",
                                "wp7_directed_lifecycle_crash_dev_v3",RACE_V4)
                or type(self.soft_corridor_m) not in (int, float) or self.soft_corridor_m != 2.
                or type(self.hard_corridor_m) not in (int, float) or self.hard_corridor_m != 4.
                or self.hard_violation != ("terminate_valid_failure" if self.owner_terminal else "reject_invalid_rollout")
                or self.gate_orientation != "authored_local_positive_z_v1"):
            raise ValueError("unregistered WP7 race contract or changed owner limits")
        if self.owner_terminal:
            owner_outcome_contract()

    @property
    def owner_terminal(self) -> bool:
        return self.version in ('wp7_directed_grounded_owner_v2','wp7_directed_lifecycle_owner_v2',
                                'wp7_directed_lifecycle_crash_dev_v3',RACE_V4)

    def provenance(self, track: TrackWorld, route: AuthoredRoute) -> dict[str, object]:
        course = {
            "navigation": asdict(route.navigation),
            "checkpoints": [asdict(c) for c in sorted(track.checkpoints, key=lambda c: c.order)],
            "finish": asdict(track.finish_lines[0]),
        }
        document = {"configuration": asdict(self), "course": course}
        if self.owner_terminal:
            document['outcome_contract']=owner_outcome_contract()
        digest = sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()
        if self.version=='wp7_directed_lifecycle_crash_dev_v3':
            document['development_crash_contract']='wp3_grounded_applied_crash_v4_dev_not_science_approved'
            digest=sha256(json.dumps(document,sort_keys=True,separators=(",", ":"),
                                     allow_nan=False).encode('utf-8')).hexdigest()
        if self.version==RACE_V4:
            document['physical_crash_contract']=descriptor()
            digest=sha256(json.dumps(document,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()
        return {**document, "sha256": digest}

    def supports_physics_mode(self, mode: str) -> bool:
        if self.version in ("wp7_directed_lifecycle_research_v1","wp7_directed_lifecycle_owner_v2",
                               "wp7_directed_lifecycle_crash_dev_v3",RACE_V4):
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
    terminal_failure: bool = False
    terminal_time_s: float | None = None
    valid_gate_events: tuple[CheckpointEntered | FinishLineEntered, ...] = ()
    terminal_reason: str | None = None


def advance_directed_race(
    track: TrackWorld, route: RaceRoute, authored: AuthoredRoute,
    progress: RaceProgressState, snapshot: SimulationSnapshot, config: WP7RaceConfig | None = None,
) -> DirectedRaceStep:
    lifecycle = config is not None and config.version in ('wp7_directed_lifecycle_research_v1','wp7_directed_lifecycle_owner_v2','wp7_directed_lifecycle_crash_dev_v3',RACE_V4)
    owner = config is not None and config.owner_terminal
    impact=snapshot.physical_crash
    if impact is not None and (config is None or config.version not in ('wp7_directed_lifecycle_crash_dev_v3',RACE_V4)):
        raise RaceContractError('physical crash requires distinct versioned development route contract')
    timed = lifecycle or owner
    if timed:
        audit = snapshot.physics_transition
        segments = snapshot.center_segments
        if (audit is None or not config.supports_physics_mode(audit.physical_mode)
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
    failure_time = None
    failure_reason = None
    peak = 0.
    soft = False
    gates = tuple((c.volume, CheckpointEntered(c.id, c.order), c.id) for c in track.checkpoints)
    gates += ((track.finish_lines[0].volume, FinishLineEntered(route.finish.id), route.finish.id),)
    for segment in segments:
        if impact is not None and segment.start_time_s>=impact.tick_contact_time_s:
            break  # Any later gate or reward is outside the certified prefix.
        if owner:
            try:
                fraction=first_hard_route_exit(authored,segment.start,segment.end)
            except ValueError as exc:
                raise RaceContractError(str(exc)) from exc
            limit=segment.end_time_s if fraction is None else segment.start_time_s+fraction*(segment.end_time_s-segment.start_time_s)
            if fraction is not None:
                failure_time=limit
                failure_reason='hard_route_violation'
            if impact is not None and impact.tick_contact_time_s<=limit+1e-10:
                limit=impact.tick_contact_time_s
                failure_time=limit
                failure_reason='physical_crash'
            points=(segment.start,segment.position_at((segment.start_time_s+limit)/2.),segment.position_at(limit))
            sampled=max(authored.horizontal_distance(p) for p in points)
            peak=max(peak,sampled)
            soft |= sampled>2.
        else:
            limit=segment.end_time_s
            corridor = screen_authored_route_sweep(authored, segment.start, segment.end)
            if not corridor.safe_for_route:
                raise RaceContractError(f"WP7 route {corridor.status}; rollout invalid")
            peak = max(peak, corridor.max_sampled_deviation_m)
            soft |= corridor.soft_zone_encountered
        for volume, event, name in gates:
            if timed:
                times = segment.directed_crossings(volume)
            else:
                crossing = directed_center_gate_crossing(segment.start,segment.end,volume)
                times = (segment.start_time_s+crossing.fraction,) if crossing is not None else ()
            for crossing_time in times:
                if crossing_time<=limit:
                    crossings.append((crossing_time, event, name))
        if failure_time is not None:
            break
    crossings.sort(key=lambda row: row[0])
    if not owner and any(abs(b[0]-a[0]) <= 1e-9 for a, b in zip(crossings, crossings[1:])):
        raise RaceContractError("simultaneous gate order is ambiguous; rollout invalid")
    current = progress
    completed = []
    valid=[]
    terminal_time=failure_time
    for index,(crossing_time, event, name) in enumerate(crossings):
        if owner and index+1<len(crossings) and abs(crossings[index+1][0]-crossing_time)<=1e-9:
            raise RaceContractError('simultaneous gate order is ambiguous; rollout invalid')
        valid.append((crossing_time,event,name))
        result = advance_race_progress(route, current, (event,))
        current = result.state
        if result.completed_checkpoint_id is not None:
            completed.append(result.completed_checkpoint_id)
        if owner and current.finished:
            terminal_time=crossing_time
            break
    if owner:
        crossings=valid
    result = RaceProgressResult(
        current, route.target_for(current), completed[-1] if completed else None,
        current.finished and not progress.finished,
    )
    return DirectedRaceStep(result, tuple(completed), tuple(c[2] for c in crossings), peak, soft,
                            tuple(c[0] for c in crossings) if timed else (),
                            owner and failure_time is not None and not current.finished,
                            terminal_time if owner else None,tuple(c[1] for c in crossings) if owner else (),
                            failure_reason if owner else None)
