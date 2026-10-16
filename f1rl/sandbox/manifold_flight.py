"""Opt-in event-time four-wheel manifold landing / certified rebound continuation.

This connects the independent impulse solver to a *flat* 2.5D ballistic
flight. Only a well-separated first event with an auditable contact set is
accepted; uncertain subsequent wheel/road contact fails closed.
"""
from __future__ import annotations

from math import acos, cos, sin
from dataclasses import replace

from .center_trajectory import CenterSegment

from .angular_landing_impulse import AngularLandingImpulsePolicy, ImpulseResolutionError
from .contact_manifold import (
    ManifoldResolutionError, WheelManifoldContact, solve_wheel_contact_manifold,
)
from .geometry.vector import Vec3
from .geometry.rotation import Quaternion
from .independent_suspension_dynamics import SuspensionDynamicsState, SuspensionResolutionError, _horizontal_yaw
from .multi_surface_support import _within_shape, _body_corners, certify_multisurface_support
from .rotating_touchdown import first_rotating_wheel_touchdown
from .swept_surface_clearance import certify_swept_chassis_clearance

_UP=Vec3(0.,1.,0.)


def advance_manifold_flight(
    airborne, flight_policy, body, atlas, obstacles, dt, gravity, dynamics,
    *, _depth=0,
):
    """Return either certified grounded state or fully consumed rebound flight.

    Flight is constant-world-omega, not yet Euler torque-free dynamics.
    Restitution-bound rebound is accepted only when every wheel is guaranteed
    to separate from the surface over the remaining timestep. A second impact
    requires an unimplemented time-ordered multi-event solver -> reject.
    """
    from .ballistic_flight import (
        FlightPhysicsError, FlightStep, FlightState, FlightTransition,
        _certify_obstacle_arc, _flight_pose, _stationary_landing_attitude,
    )
    policy=flight_policy.manifold
    if dynamics is None or dynamics.contact!=flight_policy.contact:
        raise FlightPhysicsError("manifold requires matching registered chassis inertia")
    patches, polygons=atlas
    if policy.coplanar_flat_contacts and (
        any(p.normal != _UP for p in patches)
        or len({p.origin.y for p in patches}) != 1
    ):
        raise FlightPhysicsError("coplanar contact requires exactly horizontal equal-height planes")
    if any(p.normal.y < .5 for p in patches):
        raise FlightPhysicsError("rotating manifold requires upward 2.5D road normals")
    if airborne.angular_velocity.magnitude*dt/flight_policy.collision_segments>.1:
        raise FlightPhysicsError("manifold angular sweep resolution insufficient")
    sampler=lambda t:_flight_pose(airborne,t,gravity)
    event_policy=AngularLandingImpulsePolicy(
        event_bracket_segments=64,event_bisections=32,
        event_time_tolerance_s=flight_policy.touchdown_time_epsilon_s,
    )
    try:
        first=first_rotating_wheel_touchdown(
            sampler,flight_policy.contact.wheels,atlas,dt,gravity,event_policy,
            certify_first_root=(policy.coplanar_flat_contacts or dynamics.angular_rate_mapping=="yaw_rx_rz_v1"),
        )
    except ImpulseResolutionError as exc:
        raise FlightPhysicsError(f"manifold time-of-contact unresolved: {exc}") from exc
    impact_time=first.time_s if first is not None else dt
    try:
        certify_swept_chassis_clearance(
            sampler,0.,impact_time,body,atlas,
            max_subdivisions=12,gravity_mps2=gravity,
        )
    except SuspensionResolutionError as exc:
        raise FlightPhysicsError(f"manifold body/road sweep unresolved: {exc}") from exc
    poses=_certify_obstacle_arc(
        airborne,impact_time,gravity,body,obstacles,
        flight_policy.collision_segments,
    )
    if first is None:
        return FlightStep(poses[-1],FlightState("airborne"),poses,(),0.,
                          center_segments=(CenterSegment(0.,dt,airborne.position,poses[-1].position,
                                                         airborne.velocity,gravity),))
    incoming=sampler(impact_time)
    impact_road=next(p for p in patches if p.id==first.patch_id)
    contacts=[]
    for i,local in enumerate(flight_policy.contact.wheels.positions):
        point=incoming.position+incoming.orientation.rotate(local)
        candidates = sorted((
            patch for patch, poly in zip(patches, polygons, strict=True)
            if (policy.coplanar_flat_contacts or patch.id == impact_road.id)
            and _within_shape(point.x,point.z,poly)
        ), key=lambda patch: patch.id)
        if not candidates:
            continue
        road = candidates[0]
        gap=(point-road.origin).dot(road.normal)
        if gap < -policy.simultaneous_gap_m:
            raise FlightPhysicsError("unresolved wheel penetration at manifold event")
        if abs(gap)<=policy.simultaneous_gap_m:
            contacts.append(WheelManifoldContact(
                i,road.id,local,road.normal,
            ))
    if first.wheel_index not in {c.wheel_index for c in contacts}:
        raise FlightPhysicsError("first wheel event missing from manifold")
    for other_patch,poly in zip(patches,polygons,strict=True):
        if other_patch.id == impact_road.id or policy.coplanar_flat_contacts:
            continue
        for local in flight_policy.contact.wheels.positions:
            p=incoming.position+incoming.orientation.rotate(local)
            if _within_shape(p.x,p.z,poly):
                if (p-other_patch.origin).dot(other_patch.normal)<=policy.simultaneous_gap_m:
                    raise FlightPhysicsError("multi-surface simultaneous contact unsupported")
    up=incoming.orientation.rotate(_UP)
    tilt=acos(max(-1.,min(1.,up.dot(_UP))))
    if tilt>flight_policy.max_touchdown_tilt_rad:
        raise FlightPhysicsError("manifold touchdown exceeds registered tilt")
    closing_speeds=[
        -(incoming.velocity+incoming.angular_velocity.cross(
          incoming.orientation.rotate(c.wheel_local))).dot(c.normal)
        for c in contacts
    ]
    if max(closing_speeds,default=0.)>flight_policy.max_touchdown_vertical_speed_mps:
        raise FlightPhysicsError("manifold contact closing speed unsafe")
    try:
        post,audit=solve_wheel_contact_manifold(
            incoming,tuple(contacts),dynamics,policy,
        )
    except ManifoldResolutionError as exc:
        raise FlightPhysicsError(f"manifold impulse invalid: {exc}") from exc
    rest=dt-impact_time
    if impact_time<=flight_policy.touchdown_time_epsilon_s:
        raise FlightPhysicsError("impact sequence has insufficient event-time progress")
    incoming_path = (CenterSegment(0.,impact_time,airborne.position,incoming.position,
                                   airborne.velocity,gravity),)
    # A multi-contact rebound remains airborne. Require strict monotonic
    # separation for every wheel across the entire remaining tick.
    if policy.restitution>0.:
        # Multiple impacts are opt-in, bounded in count and ordered in time.
        # Recursive flight must independently certify every intervening
        # road/body/obstacle trajectory and cannot fake a grounded interval.
        if _depth + 1 < policy.max_impacts_per_step and rest>1e-8:
            following=advance_manifold_flight(
                post,flight_policy,body,atlas,obstacles,rest,gravity,dynamics,
                _depth=_depth+1,
            )
            ordered=(FlightTransition("rebound",impact_time,impact_road.id),)
            shifted=tuple(
                FlightTransition(t.kind,impact_time+t.time_s,t.patch_id)
                for t in following.transitions
            )
            return FlightStep(
                following.vehicle,following.flight,
                (*poses[:-1],post,*following.poses[1:]),
                (*ordered,*shifted),following.remaining_dt_s,
                following.touchdown_speed_mps,following.landing_suspension,
                following.impulse_audit,following.manifold_audit,
                (audit,*following.manifold_history),
                center_segments=(*incoming_path,*(replace(s,start_time_s=s.start_time_s+impact_time,
                    end_time_s=s.end_time_s+impact_time) for s in following.center_segments)),
            )
        if len(patches)>1 and not policy.coplanar_flat_contacts:
            raise FlightPhysicsError("rebound with mixed road patches not certified")
        # A fully specified rebound requires every wheel to be initially
        # separating with a positive acceleration bound margin. Otherwise
        # refuse and defer to a future multi-event contact integrator.
        for local in flight_policy.contact.wheels.positions:
            arm=post.orientation.rotate(local)
            normal_v=(post.velocity+post.angular_velocity.cross(arm)).dot(impact_road.normal)
            # Conservatively bound the change in wheel vertical velocity
            # by gravity and angular-direction curvature.
            lower=normal_v-(gravity*impact_road.normal.y
                            +post.angular_velocity.magnitude**2*arm.magnitude)*rest
            if lower <= policy.rebound_separation_mps:
                raise FlightPhysicsError("rebound separation cannot be certified for whole remainder")
        continuation=lambda t:_flight_pose(post,t,gravity)
        try:
            certify_swept_chassis_clearance(
                continuation,0.,rest,body,atlas,
                max_subdivisions=12,gravity_mps2=gravity,
            )
        except SuspensionResolutionError as exc:
            raise FlightPhysicsError(f"rebound body sweep unresolved: {exc}") from exc
        if rest>0.:
            follow=_certify_obstacle_arc(
                post,rest,gravity,body,obstacles,
                flight_policy.collision_segments,
            )
            final=follow[-1]
            full_poses=(*poses[:-1],post,*follow[1:])
        else:
            final=post
            full_poses=(*poses[:-1],post)
        return FlightStep(
            final,FlightState("airborne"),full_poses,
            (FlightTransition("rebound",impact_time,impact_road.id),),
            0.,max(closing_speeds),None,None,audit,(audit,),
            center_segments=(*incoming_path,*((CenterSegment(impact_time,dt,post.position,final.position,
                post.velocity,gravity),) if rest > 0. else ())),
        )
    # A single event cannot convert to grounded if another wheel was not
    # actually supported or if yaw velocity cannot be represented by the
    # existing 3-DOF grounded model.
    grounded=type(post)(
        post.position,post.velocity,post.orientation,post.angular_velocity,True,
    )
    support=certify_multisurface_support(
        grounded,grounded,body,flight_policy.contact,tuple(patches),
    )
    if not support.valid:
        raise FlightPhysicsError("post-manifold four-wheel support uncertified")
    yaw=_horizontal_yaw(post.orientation)
    conjugate=Quaternion(yaw.w,-yaw.x,-yaw.y,-yaw.z)
    wlocal=conjugate.rotate(post.angular_velocity)
    orientation_state=_stationary_landing_attitude(
        post.orientation,post.position.y,
    )
    roll_rate=wlocal.z/cos(orientation_state.pitch_x_rad)
    yaw_rate=(wlocal.y+sin(orientation_state.pitch_x_rad)*roll_rate
              if dynamics.angular_rate_mapping=="yaw_rx_rz_v1" else wlocal.y)
    if abs(yaw_rate)>policy.max_ground_yaw_radps:
        raise FlightPhysicsError("grounded yaw dynamics not accepted for manifold transition")
    suspension=SuspensionDynamicsState(
        post.position.y,post.velocity.y,
        orientation_state.pitch_x_rad,wlocal.x,
        orientation_state.roll_z_rad,roll_rate,
    )
    return FlightStep(
        grounded,FlightState("grounded"),(*poses[:-1],grounded),
        (FlightTransition("landing",impact_time,impact_road.id),),
        max(0.,rest),max(closing_speeds),suspension,None,audit,(audit,),
        center_segments=incoming_path,
    )
