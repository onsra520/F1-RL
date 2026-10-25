"""Opt-in v4 fully applied rigid-face crash candidate with fail-closed timing.

This creates an accepted post-impulse VehicleState at a witnessed contact and
a stationary tail for absorbing-terminal timing. It is NOT the owner-approved
scientific scorer; science admission remains a separate, explicit gate.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

from .accepted_contact_transition import (
    AppliedContactState,apply_certified_face_impulse,absorb_crash_center_prefix
)
from .body_collision import VehicleBodyGeometry,sweep_body_against_obstacle
from .certified_edge_impact import apply_certified_edge_impulse
from .center_trajectory import CenterSegment
from .rigid_body_impact_response import RigidBodyImpactMass
from .vehicle import VehicleState


@dataclass(frozen=True,slots=True)
class PhysicalCrash:
    obstacle_id:str
    tick_contact_time_s:float
    last_proven_clear_s:float
    first_overlap_s:float
    segment_index:int
    contact:AppliedContactState
    prefix:tuple[CenterSegment,...]
    version:str="wp3_grounded_applied_crash_v4_dev"

    def __post_init__(self):
        if (not isinstance(self.obstacle_id,str) or not self.obstacle_id
            or not isinstance(self.contact,AppliedContactState)
            or not ((self.contact.kind=="obstacle_face"
                     and self.version=="wp3_grounded_applied_crash_v4_dev"
                     and not self.obstacle_id.startswith('road-patch:'))
                    or (self.contact.kind=="obstacle_edge"
                     and self.version=="wp3_grounded_applied_edge_crash_v4_dev"
                     and not self.obstacle_id.startswith('road-patch:'))
                    or (self.contact.kind=="bounded_road_body"
                     and self.version=="wp3_grounded_applied_road_crash_v4_dev"
                     and self.obstacle_id.startswith('road-patch:'))
                    or (self.contact.kind=="wheel_hard_stop"
                     and self.version=="wp3_grounded_wheel_hard_stop_crash_v4_dev"
                     and self.obstacle_id.startswith('wheel-hard-stop:')))
            or type(self.segment_index) is not int or self.segment_index<0
            or not all(type(t) in (float,int) and isfinite(t)
                   for t in (self.tick_contact_time_s,self.last_proven_clear_s,self.first_overlap_s))
            or not 0.<=self.last_proven_clear_s<self.first_overlap_s<=self.tick_contact_time_s
            or not self.prefix or self.prefix[-1].end!=self.contact.postimpact.position):
            raise ValueError("physical crash timing, accepted impulse or prefix missing")


def first_applied_grounded_crash(
    poses:tuple[VehicleState,...],
    segments:tuple[CenterSegment,...],
    obstacles:tuple,
    body:VehicleBodyGeometry,
    mass:RigidBodyImpactMass,
    *,
    fixed_dt_s:float,
)->PhysicalCrash|None:
    if len(poses)!=len(segments)+1 or not segments:
        raise ValueError("collision source must contain audited substep poses and timing")
    candidates=[]
    for obstacle in obstacles:
        for i,(p,q,segment) in enumerate(zip(poses[:-1],poses[1:],segments,strict=True)):
            verdict=sweep_body_against_obstacle(p,q,body,obstacle.volume)
            if verdict.uncertain:
                raise ValueError("obstacle collision interval cannot be certified")
            if not verdict.contact:
                continue
            if segment.initial_velocity is not None or not p.grounded or not q.grounded:
                raise ValueError("airborne obstacle impact requires a distinct ballistic solver")
            duration=segment.end_time_s-segment.start_time_s
            try:
                accepted=apply_certified_face_impulse(
                    p,q,body,obstacle.volume,mass,duration_s=duration)
                crash_version="wp3_grounded_applied_crash_v4_dev"
            except ValueError as face_error:
                # Face normal absent: ONLY a separately source-certified
                # unique SAT edge/edge witness may admit the transition.
                # Invalid or ambiguous corners still fail closed.
                try:
                    accepted=apply_certified_edge_impulse(
                        p,q,body,obstacle.volume,mass,duration_s=duration)
                except ValueError as edge_error:
                    raise ValueError(
                        f"no physical face or unique edge impulse: "
                        f"{face_error}; {edge_error}") from edge_error
                crash_version="wp3_grounded_applied_edge_crash_v4_dev"
            abs_t=segment.start_time_s+accepted.witnessed_contact_s
            shortened=absorb_crash_center_prefix(segments,abs_t,
                fixed_dt_s=fixed_dt_s,accepted=accepted)
            candidates.append(PhysicalCrash(
                obstacle.id,abs_t,segment.start_time_s+accepted.first_clear_s,
                abs_t,i,accepted,shortened,version=crash_version))
            break
    if not candidates:
        return None
    candidates.sort(key=lambda c:c.tick_contact_time_s)
    if len(candidates)>1 and candidates[1].last_proven_clear_s<=candidates[0].first_overlap_s:
        raise ValueError("simultaneous obstacle contact normals cannot be uniquely ordered")
    return candidates[0]
