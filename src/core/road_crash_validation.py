"""Fail-closed development road-impact transition source proof.

This validates exact source pose/leaf chain and recomputes the first finite
road OBB contact and rigid normal impulse. It never grants scientific Eval.
"""
from __future__ import annotations
from math import isclose,isfinite,ulp
from .physics import PhysicsTransitionError
from ..simulation.accepted_contact_transition import _apply_mass_impulse
from ..simulation.accepted_road_body_contact import apply_first_road_body_contact
from ..simulation.center_trajectory import CenterSegment
from ..simulation.geometry.vector import Vec3
from ..simulation.rigid_body_impact_response import RigidBodyImpactMass


def same_reconstructed_road_pose(a,b):
    """Source-reconstructed pose, allowing only finite IEEE quaternion roundoff.

    Spatial RK4 normalizes the quaternion again when building the impact
    source. In rare rotations that differs by one ULP in orientation and
    angular velocity, not by a distinct physical pose or impulse. Preserve
    EXACT accepted position/velocity/phase and a 1e-12 angular bound.
    """
    if (a.position!=b.position or a.velocity!=b.velocity
            or a.grounded!=b.grounded
            or (a.angular_velocity-b.angular_velocity).magnitude>1e-12):
        return False
    q,r=a.orientation,b.orientation
    return all(isfinite(x) and isfinite(y) and
               abs(x-y)<=2.*max(ulp(x),ulp(y))
               for x,y in zip((q.w,q.x,q.y,q.z),
                              (r.w,r.x,r.y,r.z),strict=True))


def verify_development_road_crash(
    result,previous,config,*,parameters,dt:float
)->None:
    crash=result.crash
    src=result.road_crash_source
    if crash is None or src is None:
        raise PhysicsTransitionError("road crash evidence absent")
    c=crash.contact
    if c.kind in ('wheel_hard_stop','wheel_edge_hard_stop'):
        return verify_development_wheel_hard_stop_crash(
            result,previous,config,parameters=parameters,dt=dt)
    airborne=not c.preimpact.grounded
    expected_version=('wp3_airborne_applied_road_crash_v4_dev' if airborne
        else 'wp3_grounded_applied_road_crash_v4_dev')
    if (config.ground_motion_version!='spatial_road_edge_partial_lifecycle_v4'
        or config.mode!='wp7_lifecycle_v1'
        or c.kind!='bounded_road_body'
        or crash.version!=expected_version
        or result.precrash_result is not None
        or result.events or result.flight_telemetry is not None
        or result.spatial_ground_telemetry is not None
        or c.postimpact.grounded!=c.preimpact.grounded
        or src.first.grounded!=c.preimpact.grounded
        or src.last.grounded!=c.preimpact.grounded
        or result.state.flight.phase!=('airborne' if airborne else 'grounded')):
        raise PhysicsTransitionError("unregistered road impact lifecycle")
    mass=RigidBodyImpactMass(config.independent_suspension_policy.mass_kg,
        Vec3(config.independent_suspension_policy.pitch_inertia_kg_m2,
             config.spatial_contact_policy.yaw_inertia_kg_m2,
             config.independent_suspension_policy.roll_inertia_kg_m2))
    poses,times,leaves=src.accepted_poses,src.accepted_times_s,src.accepted_leaves
    ballistic_leaves=src.accepted_ballistic_leaves
    brackets=src.accepted_phase_brackets
    # A v4 tick may first integrate physically accepted airborne/landing
    # leaves, then encounter the road. Preserve and independently verify
    # that phase prefix, rather than assuming every pose was grounded.
    phases=tuple((a,b,'landing' if q.grounded else 'takeoff')
        for a,b,p,q in zip(times[:-1],times[1:],poses[:-1],poses[1:])
        if p.grounded!=q.grounded)
    if (len(ballistic_leaves)!=len(leaves)
        or any(type(flag) is not bool for flag in ballistic_leaves)
        or poses[-1].grounded!=c.preimpact.grounded
        or any(flag!=(not p.grounded and not q.grounded)
            for flag,p,q in zip(ballistic_leaves,poses[:-1],poses[1:],strict=True))
        or tuple((b.lower_time_s,b.upper_time_s,b.kind) for b in brackets)!=phases):
        raise PhysicsTransitionError("road crash phase/ballistic prefix unbound")
    source_checks={
        "mass":mass!=src.mass,
        "body":src.body!=config.body_geometry,
        "road_atlas":not src.road_atlas or not src.road_atlas[0],
        "segment_start":not 0.<=src.segment_start_s<dt,
        "segment_duration":not 0.<src.segment_duration_s<=dt-src.segment_start_s+1e-12,
        "chain_lengths":len(poses)!=len(times) or len(leaves)!=len(poses)-1,
        "initial_pose":poses[0]!=previous.vehicle,
        "final_pose":not same_reconstructed_road_pose(poses[-1],src.first),
        "initial_time":times[0]!=0.,
        "final_time":abs(times[-1]-src.segment_start_s)>1e-10,
        "final_phase":poses[-1].grounded!=c.preimpact.grounded,
    }
    if any(source_checks.values()):
        faults=",".join(key for key,value in source_checks.items() if value)
        if source_checks["final_pose"]:
            a,b=poses[-1],src.first
            faults+=(f" pos_eq={a.position==b.position}"
                f" vel_eq={a.velocity==b.velocity}"
                f" quat_eq={a.orientation==b.orientation}"
                f" grounded_eq={a.grounded==b.grounded}"
                f" angular_delta={(a.angular_velocity-b.angular_velocity).magnitude:.12g}"
                f" pos_delta={(a.position-b.position).magnitude:.12g}"
                f" vel_delta={(a.velocity-b.velocity).magnitude:.12g}"
                f" q_a={a.orientation} q_b={b.orientation}")
        raise PhysicsTransitionError("road crash accepted RK4 prefix unbound: "+faults)
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
            src.road_atlas,mass,duration_s=src.segment_duration_s,
            ballistic_gravity_mps2=parameters.gravity if airborne else 0.)
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
    prefix=tuple(CenterSegment(
        a,b,p.position,q.position,p.velocity if air else None,
        parameters.gravity if air else 0.)
        for a,b,p,q,air in zip(
            times[:-1],times[1:],poses[:-1],poses[1:],ballistic_leaves,strict=True))
    prefix+=(CenterSegment(src.segment_start_s,crash.tick_contact_time_s,
        src.first.position,c.preimpact.position,
        src.first.velocity if airborne else None,
        parameters.gravity if airborne else 0.),)
    if crash.tick_contact_time_s<dt:
        prefix+=(CenterSegment(crash.tick_contact_time_s,dt,
            c.preimpact.position,c.preimpact.position),)
    if (result.center_segments!=prefix or crash.prefix!=prefix
        or result.state.vehicle!=c.postimpact
        or result.state.score!=previous.score
        or result.state.track!=previous.track
        or result.state.interactions!=previous.interactions
        or result.state.flight.phase!=('airborne' if airborne else 'grounded')):
        raise PhysicsTransitionError("road crash absorbing prefix or accepted state mismatch")


def verify_development_wheel_hard_stop_crash(
    result,previous,config,*,parameters,dt:float
)->None:
    """Independently reapply first source-witnessed wheel hard-stop impulse.

    This development terminal owns a first hard-stop, not a fictitious
    obstacle face or a subsequent unvalidated compliant contact. It consumes
    the exact source prefix then absorbs the remainder of the tick.
    """
    from ..simulation.accepted_contact_transition import (
        apply_bump_impulse_at_certified_clear_pose,
    )
    from ..simulation.wheel_bump_stop_impact import certify_wheel_bump_stop
    from ..simulation.track.patch import PlanarSurfacePatch
    crash=result.crash
    src=result.road_crash_source
    if crash is None or src is None:raise PhysicsTransitionError("wheel stop source missing")
    c=crash.contact
    is_edge=c.kind=='wheel_edge_hard_stop'
    m=RigidBodyImpactMass(config.independent_suspension_policy.mass_kg,
       Vec3(config.independent_suspension_policy.pitch_inertia_kg_m2,
            config.spatial_contact_policy.yaw_inertia_kg_m2,
            config.independent_suspension_policy.roll_inertia_kg_m2))
    local=src.wheel_hard_stop_local
    road=src.wheel_hard_stop_patch
    clear=src.wheel_hard_stop_clear_s
    witness=src.wheel_hard_stop_witness_s
    leaf=src.wheel_hard_stop_impact_audit
    poses,times,leaves=src.accepted_poses,src.accepted_times_s,src.accepted_leaves
    ballistic=src.accepted_ballistic_leaves
    if (config.mode!='wp7_lifecycle_v1'
        or config.ground_motion_version!='spatial_road_edge_partial_lifecycle_v4'
        or crash.version!=('wp3_grounded_wheel_edge_stop_crash_v4_dev' if is_edge
                            else 'wp3_grounded_wheel_hard_stop_crash_v4_dev')
        or c.kind!=('wheel_edge_hard_stop' if is_edge else 'wheel_hard_stop')
        or not isinstance(local,Vec3)
        or not isinstance(road,PlanarSurfacePatch)
        or not crash.obstacle_id==f'{"wheel-edge-stop" if is_edge else "wheel-hard-stop"}:{road.id}'
        or not isinstance(leaf,object) or leaf is None
        or c.preimpact.position!=c.postimpact.position
        or not c.preimpact.grounded or not c.postimpact.grounded
        or result.precrash_result is not None or result.events
        or result.spatial_ground_telemetry is not None or result.flight_telemetry is not None
        or m!=src.mass or src.body!=config.body_geometry
        or (is_edge and (
            not isinstance(src.wheel_edge_polygon,tuple)
            or type(src.wheel_edge_index) is not int
            or not src.road_atlas or road not in src.road_atlas[0]
            or src.road_atlas[1][src.road_atlas[0].index(road)]!=src.wheel_edge_polygon))
        or (not is_edge and (src.wheel_edge_polygon is not None
                              or src.wheel_edge_index is not None))
        or not 0.<=src.segment_start_s<dt
        or not 0.<src.segment_duration_s<=dt-src.segment_start_s+1e-12
        or len(poses)!=len(times) or len(leaves)!=len(poses)-1
        or len(ballistic)!=len(leaves)
        or poses[0]!=previous.vehicle
        or not same_reconstructed_road_pose(poses[-1],src.first)
        or times[0]!=0. or abs(times[-1]-src.segment_start_s)>1e-10
        or not src.first.grounded
        or type(clear) not in (int,float) or type(witness) not in (int,float)
        or not 0.<clear<witness<=src.segment_duration_s
        or not 0.<witness-clear<=1.0001e-10
        or crash.segment_index!=len(leaves)
        or abs(crash.tick_contact_time_s-(src.segment_start_s+witness))>1e-12
        or abs(crash.last_proven_clear_s-(src.segment_start_s+clear))>1e-12
        or crash.first_overlap_s!=crash.tick_contact_time_s
        or leaf.start_time_s!=src.segment_start_s
        or abs(leaf.end_time_s-(src.segment_start_s+clear))>1e-12
        or abs(leaf.energy_residual_j)>leaf.energy_tolerance_j
        or leaf.linear_impulse_residual_ns>1e-7
        or leaf.angular_impulse_residual_nms>1e-7):
        raise PhysicsTransitionError('wheel hard-stop source/impact prefix unbound')
    for a,b,p,q,signed,step in zip(
        times[:-1],times[1:],poses[:-1],poses[1:],ballistic,leaves,
        strict=True):
        if (not 0.<=a<b<=src.segment_start_s
            or step.start_time_s!=a or step.end_time_s!=b
            or abs(step.energy_residual_j)>step.energy_tolerance_j
            or step.linear_impulse_residual_ns>1e-7
            or step.angular_impulse_residual_nms>1e-7
            or ((q.velocity-p.velocity)*m.mass_kg-step.force_impulse_ns).magnitude>1e-7
            or (m.world_inertia(q.orientation,q.angular_velocity)
               -m.world_inertia(p.orientation,p.angular_velocity)
               -step.torque_impulse_nms).magnitude>1e-7):
            raise PhysicsTransitionError('unaccepted wheel stop preimpact RK4 leaf')
    if (((c.postimpact.velocity-src.first.velocity)*m.mass_kg-
         leaf.force_impulse_ns).magnitude>1e-7
        or (m.world_inertia(c.postimpact.orientation,c.postimpact.angular_velocity)
          -m.world_inertia(src.first.orientation,src.first.angular_velocity)
          -leaf.torque_impulse_nms).magnitude>1e-7):
        raise PhysicsTransitionError('wheel stop impulse fails accepted prefix momenta')
    try:
        if is_edge:
            from ..simulation.wheel_edge_overtravel_impact import (
                incoming_edge,apply_edge_overtravel_impulse_at_prefix,
                entry_margin,
            )
            edge=incoming_edge(src.first,src.last,local,road,
                src.wheel_edge_polygon,
                max_bump_m=config.independent_suspension_policy.contact.max_bump_m)
            if edge!=src.wheel_edge_index:
                raise ValueError('sidewall edge index does not match source')
            if (entry_margin(c.preimpact,local,src.wheel_edge_polygon,edge)<-1e-6
                or entry_margin(c.preimpact,local,src.wheel_edge_polygon,edge)>1e-9):
                raise ValueError('unwitnessed finite tire edge time')
            replay=apply_edge_overtravel_impulse_at_prefix(
                c.preimpact,m,wheel_local=local,road=road,
                polygon=src.wheel_edge_polygon,edge_index=edge,
                max_bump_m=config.independent_suspension_policy.contact.max_bump_m,
                first_clear_s=clear,first_witness_s=witness)
            cert=None
        else:
            cert=certify_wheel_bump_stop(src.first,src.last,
                wheel_local=local,road=road,
                max_bump_m=config.independent_suspension_policy.contact.max_bump_m,
                mass=m,duration_s=src.segment_duration_s,
                resolution_s=min(1e-10,src.segment_duration_s/128.))
            replay=apply_bump_impulse_at_certified_clear_pose(
                c.preimpact,m,wheel_local=local,road=road,
                max_bump_m=config.independent_suspension_policy.contact.max_bump_m,
                first_clear_s=clear,first_witness_s=witness)
    except ValueError as error:
        raise PhysicsTransitionError('wheel hard stop cannot be independently re-certified') from error
    if ((not is_edge and (cert.first_violating_time_s<clear-1e-7
        or cert.first_clear_time_s>witness+1e-7))
        or replay!=c):
        raise PhysicsTransitionError('wheel stop source and signed impulse differ')
    prefix=tuple(CenterSegment(a,b,p.position,q.position,
               p.velocity if air else None,
               parameters.gravity if air else 0.)
         for a,b,p,q,air in zip(
            times[:-1],times[1:],poses[:-1],poses[1:],
            ballistic,strict=True))
    prefix+=(CenterSegment(src.segment_start_s,
             crash.tick_contact_time_s,src.first.position,c.preimpact.position),)
    if crash.tick_contact_time_s<dt:
        prefix+=(CenterSegment(crash.tick_contact_time_s,dt,
            c.preimpact.position,c.preimpact.position),)
    if (result.center_segments!=prefix or crash.prefix!=prefix
        or result.state.vehicle!=c.postimpact
        or result.state.score!=previous.score
        or result.state.track!=previous.track
        or result.state.interactions!=previous.interactions
        or result.state.flight.phase!='grounded'):
        raise PhysicsTransitionError('wheel stop absorbing state/reward prefix mismatched')
