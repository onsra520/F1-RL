"""Fail-closed development road-impact transition source proof.

This validates exact source pose/leaf chain and recomputes the first finite
road OBB contact and rigid normal impulse. It never grants scientific Eval.
"""
from __future__ import annotations
from math import isclose,isfinite
from .physics import PhysicsTransitionError
from ..sandbox.accepted_contact_transition import _apply_mass_impulse
from ..sandbox.accepted_road_body_contact import apply_first_road_body_contact
from ..sandbox.center_trajectory import CenterSegment
from ..sandbox.geometry.vector import Vec3
from ..sandbox.rigid_body_impact_response import RigidBodyImpactMass


def verify_development_road_crash(
    result,previous,config,*,dt:float
)->None:
    crash=result.crash
    src=result.road_crash_source
    if crash is None or src is None:
        raise PhysicsTransitionError("road crash evidence absent")
    c=crash.contact
    if (config.ground_motion_version!='spatial_road_edge_partial_lifecycle_v4'
        or config.mode!='wp7_lifecycle_v1'
        or c.kind!='bounded_road_body'
        or crash.version!='wp3_grounded_applied_road_crash_v4_dev'
        or result.precrash_result is not None
        or result.events or result.flight_telemetry is not None
        or result.spatial_ground_telemetry is not None
        or not c.preimpact.grounded or not c.postimpact.grounded):
        raise PhysicsTransitionError("unregistered road impact lifecycle")
    mass=RigidBodyImpactMass(config.independent_suspension_policy.mass_kg,
        Vec3(config.independent_suspension_policy.pitch_inertia_kg_m2,
             config.spatial_contact_policy.yaw_inertia_kg_m2,
             config.independent_suspension_policy.roll_inertia_kg_m2))
    poses,times,leaves=src.accepted_poses,src.accepted_times_s,src.accepted_leaves
    if (mass!=src.mass or src.body!=config.body_geometry
        or not src.road_atlas or not src.road_atlas[0]
        or not 0.<=src.segment_start_s<dt
        or not 0.<src.segment_duration_s<=dt-src.segment_start_s+1e-12
        or len(poses)!=len(times) or len(leaves)!=len(poses)-1
        or poses[0]!=previous.vehicle or poses[-1]!=src.first
        or times[0]!=0. or abs(times[-1]-src.segment_start_s)>1e-10
        or any(not pose.grounded for pose in poses)):
        raise PhysicsTransitionError("road crash accepted RK4 prefix unbound")
    for a,b,p,q,leaf in zip(times[:-1],times[1:],poses[:-1],poses[1:],leaves):
        if (not 0.<=a<b<=src.segment_start_s
            or leaf.start_time_s!=a or leaf.end_time_s!=b
            or not abs(leaf.energy_residual_j)<=leaf.energy_tolerance_j
            or leaf.linear_impulse_residual_ns>1e-7
            or leaf.angular_impulse_residual_nms>1e-7
            or ((q.velocity-p.velocity)*mass.mass_kg-leaf.force_impulse_ns).magnitude>1e-7
            or (mass.world_inertia(q.orientation,q.angular_velocity)
                -mass.world_inertia(p.orientation,p.angular_velocity)
                -leaf.torque_impulse_nms).magnitude>1e-7):
            raise PhysicsTransitionError("road impact prefix momentum/work not accepted")
    try:
        replay=apply_first_road_body_contact(src.first,src.last,src.body,
            src.road_atlas,mass,duration_s=src.segment_duration_s)
        checked=_apply_mass_impulse(c.preimpact,mass,
            c.impulse_n_s,c.angular_impulse_nms,
            c.kinetic_before_j,c.kinetic_after_j,c.energy_lost_j,
            first_clear=c.first_clear_s,witness=c.witnessed_contact_s,
            kind='bounded_road_body')
    except ValueError as error:
        raise PhysicsTransitionError("road impulse/source cannot be replayed") from error
    if (replay is None or checked!=c or replay.contact!=c
        or crash.obstacle_id!=f'road-patch:{replay.patch_id}'
        or crash.segment_index!=len(leaves)
        or not isclose(crash.tick_contact_time_s,
             src.segment_start_s+c.witnessed_contact_s,abs_tol=1e-12)
        or not isclose(crash.last_proven_clear_s,
             src.segment_start_s+c.first_clear_s,abs_tol=1e-12)
        or crash.first_overlap_s!=crash.tick_contact_time_s):
        raise PhysicsTransitionError("road collision signed event does not match source")
    prefix=tuple(CenterSegment(a,b,p.position,q.position)
        for a,b,p,q in zip(times[:-1],times[1:],poses[:-1],poses[1:]))
    prefix+=(CenterSegment(src.segment_start_s,crash.tick_contact_time_s,
        src.first.position,c.preimpact.position),)
    if crash.tick_contact_time_s<dt:
        prefix+=(CenterSegment(crash.tick_contact_time_s,dt,
            c.preimpact.position,c.preimpact.position),)
    if (result.center_segments!=prefix or crash.prefix!=prefix
        or result.state.vehicle!=c.postimpact
        or result.state.score!=previous.score
        or result.state.track!=previous.track
        or result.state.interactions!=previous.interactions
        or result.state.flight.phase!='grounded'):
        raise PhysicsTransitionError("road crash absorbing prefix or accepted state mismatch")
