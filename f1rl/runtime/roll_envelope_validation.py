"""Independent physics audit of source-certified model-envelope roll stop.

This is not a contact response. Any fabricated impulse, road overlap, source
time or reward prefix is rejected before the evaluator sees the episode.
"""
from __future__ import annotations
from math import isclose
from ..sandbox.roll_envelope_terminal import RollEnvelopeStop,_roll,VERSION
from ..sandbox.geometry.vector import Vec3
from ..sandbox.geometry.rotation import Quaternion
from ..sandbox.body_collision import obb_separation_lower_bound
from ..sandbox.track.volume import OrientedBoxVolume
from ..sandbox.center_trajectory import CenterSegment
from .physics import PhysicsTransitionError

def verify_roll_envelope_stop(result,previous,config,*,dt):
    e=result.roll_envelope_stop
    if not isinstance(e,RollEnvelopeStop) or e.version!=VERSION:
        raise PhysicsTransitionError("unregistered roll-envelope event")
    src=e.source
    policy=config.independent_suspension_policy
    spatial=config.spatial_contact_policy
    if (config.ground_motion_version!='spatial_road_edge_partial_lifecycle_v4'
        or config.mode!='wp7_lifecycle_v1'
        or result.crash is not None or result.events
        or result.flight_telemetry is not None
        or result.spatial_ground_telemetry is not None
        or result.suspension_telemetry is not None
        or result.precrash_result is not None
        or result.road_crash_source is not None
        or result.state.vehicle!=e.accepted
        or result.state.score!=previous.score
        or result.state.track!=previous.track
        or result.state.interactions!=previous.interactions
        or result.state.flight.phase!='grounded'
        or not isclose(e.roll_limit_rad,policy.max_abs_tilt_rad,abs_tol=1e-14)
        or not 0.<=src.segment_start_s<e.first_clear_s<e.first_outside_s<=dt
        or abs(e.first_clear_s-(src.segment_start_s+src.first_clear_local_s))>1e-12
        or abs(e.first_outside_s-(src.segment_start_s+src.first_outside_local_s))>1e-12
        or not e.first_outside_s-e.first_clear_s<=1.0001e-10
        or src.source_rejected!=e.rejected
        or src.body!=config.body_geometry
        or src.previous_poses[0]!=previous.vehicle
        or src.previous_poses[-1]!=src.source_start
        or src.previous_times_s[0]!=0.
        or abs(src.previous_times_s[-1]-src.segment_start_s)>1e-12
        or len(src.previous_poses)!=len(src.previous_times_s)
        or len(src.previous_leaves)!=len(src.previous_times_s)-1
        or len(src.road_atlas)!=2 or not src.road_atlas[0]
        or abs(_roll(src.source_start)-e.initial_roll_rad)>1e-10
        or not abs(e.initial_roll_rad)<e.roll_limit_rad
        or not 0.<e.side*e.clear_roll_rad<=e.roll_limit_rad
        or not e.side*e.rejected_roll_rad>e.roll_limit_rad
        or e.side*(_roll(e.accepted)-_roll(src.source_start))<=0.
        or not e.accepted.grounded or not e.rejected.grounded):
        raise PhysicsTransitionError("roll-envelope source, domain or chronology invalid")

    def momentum(v):
        q=v.orientation
        inv=Quaternion(q.w,-q.x,-q.y,-q.z)
        w=inv.rotate(v.angular_velocity)
        return q.rotate(Vec3(policy.pitch_inertia_kg_m2*w.x,
              spatial.yaw_inertia_kg_m2*w.y,
              policy.roll_inertia_kg_m2*w.z))

    for a,b,p,q,leaf in zip(src.previous_times_s[:-1],src.previous_times_s[1:],
                            src.previous_poses[:-1],src.previous_poses[1:],
                            src.previous_leaves,strict=True):
        if (not 0.<=a<b<=src.segment_start_s
            or leaf.start_time_s!=a or leaf.end_time_s!=b
            or abs(leaf.energy_residual_j)>leaf.energy_tolerance_j
            or leaf.linear_impulse_residual_ns>1e-7
            or leaf.angular_impulse_residual_nms>1e-7
            or ((q.velocity-p.velocity)*policy.mass_kg-leaf.force_impulse_ns).magnitude>1e-7
            or (momentum(q)-momentum(p)-leaf.torque_impulse_nms).magnitude>1e-7):
            raise PhysicsTransitionError("roll-domain accepted prefix invalid")
    leaf=src.impact_prefix_audit
    if (leaf.start_time_s!=src.segment_start_s
        or leaf.end_time_s!=e.first_clear_s
        or abs(leaf.energy_residual_j)>leaf.energy_tolerance_j
        or leaf.linear_impulse_residual_ns>1e-7
        or leaf.angular_impulse_residual_nms>1e-7
        or leaf.dissipation_j<0.
        or ((e.accepted.velocity-src.source_start.velocity)*policy.mass_kg
            -leaf.force_impulse_ns).magnitude>1e-7
        or (momentum(e.accepted)-momentum(src.source_start)
            -leaf.torque_impulse_nms).magnitude>1e-7):
        raise PhysicsTransitionError("roll-domain event invents impulse/energy")

    for state in (src.source_start,e.accepted,e.rejected):
        box=src.body.box_at(state.position,state.orientation)
        for patch in src.road_atlas[0]:
            road=OrientedBoxVolume(patch.origin,Vec3(patch.width/2.,1e-9,patch.length/2.),
                Quaternion.from_basis(patch.right,patch.normal,patch.forward))
            if obb_separation_lower_bound(box,road)<=1e-6:
                raise PhysicsTransitionError("roll-domain event is actually body-road contact")
    prefix=tuple(CenterSegment(a,b,p.position,q.position)
        for a,b,p,q in zip(src.previous_times_s[:-1],src.previous_times_s[1:],
                           src.previous_poses[:-1],src.previous_poses[1:],strict=True))
    # Earlier ballistic intervals may contain a certified initial velocity.
    if len(e.prefix)<len(prefix)+1:
        raise PhysicsTransitionError("roll-envelope prefix missing")
    for x,y in zip(prefix,e.prefix,strict=False):
        if (x.start_time_s!=y.start_time_s or x.end_time_s!=y.end_time_s
            or x.start!=y.start or x.end!=y.end):
            raise PhysicsTransitionError("roll prefix trajectory differs from source")
    last=CenterSegment(src.segment_start_s,e.first_clear_s,
        src.source_start.position,e.accepted.position)
    expected=(*e.prefix[:len(prefix)],last)
    if e.first_clear_s<dt:
        expected+=(CenterSegment(e.first_clear_s,dt,
                   e.accepted.position,e.accepted.position),)
    if e.prefix!=expected or result.center_segments!=expected:
        raise PhysicsTransitionError("roll-domain event/reward prefix does not match source")
