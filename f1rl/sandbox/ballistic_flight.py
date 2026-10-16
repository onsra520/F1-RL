"""Prospective WP7 flight kernel: explicit launch, exact ballistic arc, guarded landing.

Strict v1 domain: fixed airborne orientation, horizontal 2.5D support patches.
No automatic ground-to-air release under partial wheel contact, no slope
landing, no impulse/tire-force claims. Unproven collision => blocked rollout.
Landing consumes only its physical portion of dt; caller MUST integrate or
reject the reported remainder rather than silently changing simulation time.
"""
from __future__ import annotations

from .center_trajectory import CenterSegment

from dataclasses import dataclass
from math import acos, atan2, cos, hypot, isfinite, sqrt

from .body_collision import VehicleBodyGeometry, sweep_body_against_obstacle
from .contact_manifold import ContactManifoldPolicy, ManifoldImpulseAudit
from .torque_free_swept_ccd import (
    TorqueFreeSweepPolicy, TorqueFreeSweepResolutionError,
    certify_torque_free_flight_sweep,
)
from .torque_free_rotation import (
    TorqueFreeRotationPolicy, AngularRigidBodyState,
    AngularRigidBodyAudit, RigidBodyIntegrationError,
    advance_torque_free_rotation,
)
from .angular_landing_impulse import (
    AngularLandingImpulsePolicy, ImpulseResolutionError, WheelImpulseAudit,
    solve_one_wheel_normal_impulse,
)
from .rotating_touchdown import first_rotating_wheel_touchdown
from .dynamics.integrator import _advance_orientation
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .multi_surface_support import (
    MultiSurfaceSuspensionPolicy, _atlas, _within_shape, _body_corners,
    _intersection, certify_multisurface_support,
)
from .road_footprint import _hull
from .road_footprint import body_xz_polygon
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy,
    SuspensionDynamicsState, _horizontal_yaw, suspension_orientation,
    SuspensionResolutionError,
)
from .swept_surface_clearance import certify_swept_chassis_clearance
from .track.patch import PlanarSurfacePatch
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState

_UP = Vec3(0., 1., 0.)
_EPS = 1e-9


class FlightPhysicsError(RuntimeError):
    """Invalid/unsupported prospective flight, never a scored collision event."""


@dataclass(frozen=True, slots=True)
class BallisticFlightPolicy:
    contact: MultiSurfaceSuspensionPolicy
    max_touchdown_vertical_speed_mps: float
    max_touchdown_tilt_rad: float
    collision_segments: int = 16
    touchdown_time_epsilon_s: float = 1e-8
    max_patch_tilt_rad: float = 0.  # flat-only unless explicitly opt-in
    angular_impulse: AngularLandingImpulsePolicy | None = None
    manifold: ContactManifoldPolicy | None = None
    torque_free: TorqueFreeRotationPolicy | None = None
    torque_free_sweep: TorqueFreeSweepPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contact, MultiSurfaceSuspensionPolicy):
            raise ValueError("flight requires explicit independent contact contract")
        if self.angular_impulse is not None and not isinstance(
            self.angular_impulse, AngularLandingImpulsePolicy
        ):
            raise ValueError("angular_impulse must be a registered contact policy")
        if self.manifold is not None and not isinstance(self.manifold, ContactManifoldPolicy):
            raise ValueError("manifold must be a registered contact policy")
        if self.torque_free is not None and not isinstance(
            self.torque_free, TorqueFreeRotationPolicy
        ):
            raise ValueError("torque_free must be a registered angular integration policy")
        if self.torque_free_sweep is not None and (
            not isinstance(self.torque_free_sweep,TorqueFreeSweepPolicy)
            or self.torque_free is None
        ):
            raise ValueError("torque_free_sweep requires registered torque_free policy")
        if self.manifold is not None and self.angular_impulse is not None:
            raise ValueError("legacy single-wheel and manifold policies are mutually exclusive")
        for name in ("max_touchdown_vertical_speed_mps",
                     "max_touchdown_tilt_rad", "touchdown_time_epsilon_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not isfinite(value) or value <= 0.):
                raise ValueError(f"{name} must be finite positive")
        if (type(self.max_patch_tilt_rad) not in (int,float)
                or not isfinite(self.max_patch_tilt_rad)
                or not 0. <= self.max_patch_tilt_rad <= .35):
            raise ValueError("max_patch_tilt_rad must be finite within [0,0.35]")
        if self.max_touchdown_tilt_rad >= .75:
            raise ValueError("touchdown tilt must obey small-angle research domain")
        if type(self.collision_segments) is not int or not 2 <= self.collision_segments <= 256:
            raise ValueError("collision_segments must be integer in [2, 256]")


@dataclass(frozen=True, slots=True)
class FlightState:
    """'takeoff' requires explicit authorized release from a supported pose."""
    phase: str = "grounded"

    def __post_init__(self) -> None:
        if self.phase not in ("grounded", "takeoff", "airborne"):
            raise ValueError("flight phase must be grounded, takeoff or airborne")


@dataclass(frozen=True, slots=True)
class FlightTransition:
    kind: str
    time_s: float
    patch_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("takeoff", "landing", "rebound"):
            raise ValueError("unknown flight transition")
        if not isfinite(self.time_s) or self.time_s < 0.:
            raise ValueError("flight event time must be nonnegative")


@dataclass(frozen=True, slots=True)
class FlightStep:
    vehicle: VehicleState
    flight: FlightState
    poses: tuple[VehicleState, ...]
    transitions: tuple[FlightTransition, ...]
    remaining_dt_s: float
    touchdown_speed_mps: float | None = None
    landing_suspension: SuspensionDynamicsState | None = None
    impulse_audit: WheelImpulseAudit | None = None
    manifold_audit: ManifoldImpulseAudit | None = None
    manifold_history: tuple[ManifoldImpulseAudit, ...] = ()
    angular_audit: AngularRigidBodyAudit | None = None
    center_segments: tuple[CenterSegment, ...] = ()

    def __post_init__(self) -> None:
        if not self.poses or self.vehicle != self.poses[-1]:
            raise ValueError("flight poses must include final pose")
        if not isfinite(self.remaining_dt_s) or self.remaining_dt_s < 0.:
            raise ValueError("invalid remaining simulation time")


def authorize_takeoff(
    grounded: VehicleState, policy: BallisticFlightPolicy,
    body: VehicleBodyGeometry, surfaces: tuple[PlanarSurfacePatch, ...],
) -> VehicleState:
    """Explicit upward release only; a wheel dropping off an edge is NOT inferred."""
    if not isinstance(grounded, VehicleState) or not grounded.grounded:
        raise FlightPhysicsError("takeoff requires grounded previous pose")
    if grounded.velocity.y <= 0.:
        raise FlightPhysicsError("takeoff requires positive upward launch velocity")
    if grounded.angular_velocity.magnitude > _EPS:
        raise FlightPhysicsError("rotating airborne launch not yet supported")
    if not isinstance(policy, BallisticFlightPolicy) or not isinstance(body, VehicleBodyGeometry):
        raise ValueError("invalid flight launch configuration")
    if not certify_multisurface_support(grounded, grounded, body, policy.contact, surfaces).valid:
        raise FlightPhysicsError("takeoff has no certified four-wheel/body launch support")
    return VehicleState(
        grounded.position, grounded.velocity, grounded.orientation,
        grounded.angular_velocity, False,
    )


def _flight_pose(start: VehicleState, t: float, gravity: float) -> VehicleState:
    if t < 0. or not isfinite(t):
        raise ValueError("nonnegative finite time required")
    return VehicleState(
        start.position + start.velocity*t + Vec3(0., -.5*gravity*t*t, 0.),
        start.velocity + Vec3(0., -gravity*t, 0.),
        _advance_orientation(start.orientation, start.angular_velocity, t),
        start.angular_velocity, False,
    )


def _rotating_exit_certified(start, policy, body, patches, polygons, dt, gravity):
    """Narrow but provable one-platform, outward-only rotating flight.

    Each wheel's forward coordinate must already be outside the platform,
    and its minimum forward speed must remain strictly positive. The chassis
    must stay above the flat road over the entire interval under a bound on
    translation, gravitational sag and rotation. Otherwise reject rather than
    claiming a rotated-touchdown model which does not exist yet.
    """
    if len(patches) != 1:
        raise FlightPhysicsError("rotating flight requires one flat exit platform")
    patch, shape = patches[0], polygons[0]
    # The platform outline lives in XZ; don't use 3D forward projection
    # (which can falsely classify a wheel above an inclined road as exited).
    fx, fz = patch.forward.x, patch.forward.z
    flen = hypot(fx, fz)
    fx, fz = fx/flen, fz/flen
    horizontal_speed = start.velocity.x*fx + start.velocity.z*fz
    omega = start.angular_velocity.magnitude
    max_wheel_radius = max(w.magnitude for w in policy.contact.wheels.positions)
    if horizontal_speed <= omega*max_wheel_radius + 1e-9:
        raise FlightPhysicsError("rotating airborne path not monotonic away from road")
    for local in policy.contact.wheels.positions:
        point = start.position + start.orientation.rotate(local)
        relative = (point.x-patch.origin.x)*fx+(point.z-patch.origin.z)*fz
        if relative <= patch.length*flen/2.:
            raise FlightPhysicsError("rotating takeoff wheel still overlaps exit platform")
    # Worst-case signed-plane clearance of any underside point throughout
    # this outward-only rotating interval, including gravity and rotation.
    start_clearance = min(
        (p-patch.origin).dot(patch.normal)
        for p in _body_corners(start, body)
    )
    worst = (abs(start.velocity.dot(patch.normal))*dt
             +.5*gravity*patch.normal.y*dt*dt
             +omega*body.motion_radius_m*dt)
    if start_clearance-worst <= 1e-9:
        raise FlightPhysicsError("rotating airborne chassis clearance unresolved")


def _descending_root(height: float, vertical_velocity: float, gravity: float) -> float | None:
    if height < -_EPS:
        return None
    disc = vertical_velocity**2 + 2.*gravity*max(0., height)
    return (vertical_velocity + sqrt(disc)) / gravity


def _flat_atlas(
    surfaces: tuple[PlanarSurfacePatch, ...], policy: BallisticFlightPolicy,
):
    atlas = _atlas(surfaces, policy.contact.seam_height_tolerance_m)
    if atlas is None:
        raise FlightPhysicsError("unsupported flight road atlas or stacked decks")
    patches, polygons = atlas
    if any(atan2(hypot(p.normal.x, p.normal.z), p.normal.y)
           > policy.max_patch_tilt_rad + _EPS for p in patches):
        raise FlightPhysicsError("sloped touchdown exceeds registered flight policy")
    return patches, polygons


def _landable_events(start, dt, gravity, policy, body, patches, polygons):
    """Exact descending plane crossings for fixed-attitude wheels and OBB corners.

    Sloped-road relative gap g(t) has acceleration -gravity because the
    world-vertical acceleration projected on a plane is normalized by n.y.
    Only patch-covered crossings are eligible. Other road intrusion is
    guarded by the separate conservative full-body swept-plane verifier.
    """
    candidates = []
    body_corners = _body_corners(start, body)
    wheel_positions = tuple(start.position + start.orientation.rotate(w)
                            for w in policy.contact.wheels.positions)
    for patch, poly in zip(patches, polygons, strict=True):
        normal = patch.normal
        rel_v = start.velocity.dot(normal) / normal.y
        for vertex in body_corners:
            gap = (vertex - patch.origin).dot(normal) / normal.y
            t = _descending_root(gap, rel_v, gravity)
            if t is not None and 0. < t <= dt:
                projected = vertex + start.velocity*t + Vec3(0., -.5*gravity*t*t, 0.)
                if _within_shape(projected.x, projected.z, poly):
                    candidates.append((t, 0, patch.id, "body"))
        for i, point in enumerate(wheel_positions):
            gap = (point - patch.origin).dot(normal) / normal.y
            t = _descending_root(gap, rel_v, gravity)
            if t is not None and 0. < t <= dt:
                projected = point + start.velocity*t + Vec3(0., -.5*gravity*t*t, 0.)
                if _within_shape(projected.x, projected.z, poly):
                    candidates.append((t, 1, patch.id, str(i)))
    return sorted(candidates)


def _certify_obstacle_arc(
    start: VehicleState,
    end_s: float,
    gravity: float,
    body: VehicleBodyGeometry,
    obstacles: tuple[OrientedBoxVolume, ...],
    segments: int,
) -> tuple[VehicleState, ...]:
    """Conservative parabola envelope using y inflation g*h^2/8.

    Between chord endpoints, vertical sagitta of constant-acceleration
    ballistic motion is at most g*h²/8. Inflating obstacle OBBs vertically
    in WORLD coordinates is valid only for axis-aligned obstacle orientations.
    Non-axis-aligned boxes require a separate continuous solver.
    """
    if any(not isinstance(o, OrientedBoxVolume) for o in obstacles):
        raise ValueError("flight obstacle volumes must be OrientedBoxVolume")
    poses = tuple(_flight_pose(start, end_s*i/segments, gravity)
                  for i in range(segments+1))
    for i, (a, b) in enumerate(zip(poses, poses[1:])):
        h = end_s/segments
        sagitta = gravity*h*h/8.
        for obstacle in obstacles:
            # The sagitta displacement is along world Y. In an oriented
            # obstacle frame, its containment interval is bounded by its
            # projections onto each local unit axis (Minkowski expansion).
            q = obstacle.orientation
            ax = q.rotate(Vec3(1.,0.,0.))
            ay = q.rotate(Vec3(0.,1.,0.))
            az = q.rotate(Vec3(0.,0.,1.))
            inflated = OrientedBoxVolume(
                obstacle.center,
                Vec3(obstacle.half_extents.x+sagitta*abs(ax.y),
                     obstacle.half_extents.y+sagitta*abs(ay.y),
                     obstacle.half_extents.z+sagitta*abs(az.y)),
                q,
            )
            sweep = sweep_body_against_obstacle(a, b, body, inflated)
            if sweep.contact or sweep.uncertain:
                raise FlightPhysicsError(
                    f"ballistic obstacle sweep blocked at segment {i}: "
                    f"{'contact' if sweep.contact else 'uncertain'}"
                )
    return poses


def _stationary_landing_attitude(orientation, height):
    """Recover local pitch/roll from yaw*Rx(pitch)*Rz(roll), then verify."""
    yaw = _horizontal_yaw(orientation)
    f = orientation.rotate(Vec3(0.,0.,1.))
    yaw_f = yaw.rotate(Vec3(0.,0.,1.))
    pitch = atan2(-f.y, f.dot(yaw_f))
    right = orientation.rotate(Vec3(1.,0.,0.))
    yaw_right = yaw.rotate(Vec3(1.,0.,0.))
    if cos(pitch) <= .5:
        raise FlightPhysicsError("landing attitude outside recoverable small-angle range")
    roll = atan2(right.y / cos(pitch), right.dot(yaw_right))
    state = SuspensionDynamicsState(height, 0., pitch, 0., roll, 0.)
    reconstruction = suspension_orientation(yaw, state)
    closeness = abs(
        reconstruction.w*orientation.w + reconstruction.x*orientation.x
        + reconstruction.y*orientation.y + reconstruction.z*orientation.z
    )
    if 1. - closeness > 1e-8:
        raise FlightPhysicsError("landing orientation cannot map to suspension state")
    return state


def _rotating_impulse_flight(
    airborne,policy,body,patches,polygons,obstacles,dt,gravity,landing_dynamics,
):
    """Opt-in single wheel event; no implied multi-contact impulse acceptance."""
    option=policy.angular_impulse
    if option is None or landing_dynamics is None:
        raise FlightPhysicsError("rotating touchdown requires explicit impulse inertia policy")
    if landing_dynamics.contact != policy.contact:
        raise FlightPhysicsError("angular impulse contact geometry mismatch")
    if airborne.angular_velocity.magnitude > option.max_pre_angular_speed_radps:
        raise FlightPhysicsError("airborne rotation outside impact policy")
    if airborne.angular_velocity.magnitude*dt/policy.collision_segments > .1:
        raise FlightPhysicsError("angular sweep resolution insufficient")
    if any(abs(p.normal.y-1.) > 1e-9 for p in patches):
        raise FlightPhysicsError("sloped rotating touchdown needs yaw/contact manifold solver")
    trajectory=lambda t:_flight_pose(airborne,t,gravity)
    try:
        event=first_rotating_wheel_touchdown(
            trajectory,policy.contact.wheels,(patches,polygons),dt,gravity,option,
        )
    except ImpulseResolutionError as exc:
        raise FlightPhysicsError(f"rotating wheel event unresolved: {exc}") from exc
    end_time=event.time_s if event is not None else dt
    try:
        certify_swept_chassis_clearance(
            trajectory,0.,end_time,body,(patches,polygons),
            max_subdivisions=12,gravity_mps2=gravity,
        )
    except SuspensionResolutionError as exc:
        raise FlightPhysicsError(f"rotating body/road sweep unresolved: {exc}") from exc
    poses=_certify_obstacle_arc(
        airborne,end_time,gravity,body,obstacles,policy.collision_segments,
    )
    if event is None:
        return FlightStep(poses[-1],FlightState("airborne"),poses,(),0.)
    incoming=trajectory(end_time)
    road=next(p for p in patches if p.id==event.patch_id)
    if abs(road.normal.y-1.) > 1e-9:
        raise FlightPhysicsError("rotating impulse road must be flat")
    # A rigid-body manifold with two simultaneous wheel references requires
    # a different solver: reject even if both fit within suspension travel.
    for idx,local in enumerate(policy.contact.wheels.positions):
        if idx == event.wheel_index:
            continue
        point=incoming.position+incoming.orientation.rotate(local)
        if _within_shape(point.x,point.z,
                         polygons[patches.index(road)]):
            if abs((point-road.origin).dot(road.normal)) <= option.unique_contact_gap_m:
                raise FlightPhysicsError("simultaneous wheel contact manifold unsupported")
    wheel_local=policy.contact.wheels.positions[event.wheel_index]
    wheel_world=incoming.position+incoming.orientation.rotate(wheel_local)
    normal_velocity=(incoming.velocity+incoming.angular_velocity.cross(
        wheel_world-incoming.position
    )).dot(road.normal)
    if -normal_velocity > policy.max_touchdown_vertical_speed_mps:
        raise FlightPhysicsError("unsafe rotating wheel contact speed")
    up=incoming.orientation.rotate(_UP)
    tilt=acos(max(-1.,min(1.,up.dot(_UP))))
    if tilt > policy.max_touchdown_tilt_rad:
        raise FlightPhysicsError("rotating touchdown exceeds registered tilt")
    try:
        post,audit=solve_one_wheel_normal_impulse(
            incoming,wheel_local,event.wheel_index,event.patch_id,
            road.normal,landing_dynamics,option,
        )
    except ImpulseResolutionError as exc:
        raise FlightPhysicsError(f"angular landing impulse invalid: {exc}") from exc
    if audit.normal_speed_after_mps > 1e-7:
        raise FlightPhysicsError(
            "rebounding wheel impulse cannot transition directly to grounded"
        )
    attitude=_stationary_landing_attitude(incoming.orientation,incoming.position.y)
    yaw=_horizontal_yaw(incoming.orientation)
    q_conj=Quaternion(yaw.w,-yaw.x,-yaw.y,-yaw.z)
    angular_local=q_conj.rotate(post.angular_velocity)
    if abs(angular_local.y) > option.max_ground_yaw_rate_radps:
        raise FlightPhysicsError("post-impact yaw component beyond 3DOF ground model")
    pitch_rate=angular_local.x
    roll_rate=angular_local.z/cos(attitude.pitch_x_rad)
    if (abs(pitch_rate) > option.max_post_angular_speed_radps
            or abs(roll_rate) > option.max_post_angular_speed_radps):
        raise FlightPhysicsError("post-impact suspension rate exceeds policy")
    suspension=SuspensionDynamicsState(
        post.position.y,post.velocity.y,
        attitude.pitch_x_rad,pitch_rate,
        attitude.roll_z_rad,roll_rate,
    )
    grounded=VehicleState(
        post.position,post.velocity,post.orientation,post.angular_velocity,True,
    )
    support=certify_multisurface_support(
        grounded,grounded,body,policy.contact,tuple(patches),
    )
    if not support.valid:
        raise FlightPhysicsError(
            f"post-impulse wheel/body support invalid {support.status}: {support.reason}"
        )
    return FlightStep(
        grounded,FlightState("grounded"),(*poses[:-1],grounded),
        (FlightTransition("landing",end_time,event.patch_id),),
        max(0.,dt-end_time),-normal_velocity,suspension,audit,
    )


def _certified_high_clearance_torque_free(
    airborne, policy, body, patches, obstacles, dt, gravity, dynamics,
):
    """Use full Euler free-spin only if a sphere bounds ALL road interactions.

    Road y_max is the maximum height of the finite planar corners. Because
    vertical ballistic motion is concave down, the minimum centre altitude
    on [0,dt] is attained at an endpoint. A sphere about the centre therefore
    proves every rigid wheel/body point above every road patch for all t.
    No obstacles are allowed until certified non-SLERP rotating sweeps exist.
    """
    opts=policy.torque_free
    if dynamics is None or policy.manifold is None:
        raise FlightPhysicsError("torque-free flight requires registered 3-axis inertia")
    if (dynamics.contact!=policy.contact
            or abs(opts.pitch_inertia_kg_m2-dynamics.pitch_inertia_kg_m2)>1e-8
            or abs(opts.roll_inertia_kg_m2-dynamics.roll_inertia_kg_m2)>1e-8
            or abs(opts.yaw_inertia_kg_m2-policy.manifold.yaw_inertia_kg_m2)>1e-8):
        raise FlightPhysicsError("torque-free and manifold inertia registrations mismatch")
    if policy.torque_free_sweep is not None:
        try:
            result=certify_torque_free_flight_sweep(
                airborne,opts,body,policy.contact.wheels,
                _atlas(patches,policy.contact.seam_height_tolerance_m),
                obstacles,duration_s=dt,gravity_mps2=gravity,
                sweep=policy.torque_free_sweep,
            )
        except TorqueFreeSweepResolutionError as exc:
            raise FlightPhysicsError(f"torque-free continuous collision blocked: {exc}") from exc
        return FlightStep(
            result.vehicle,FlightState("airborne"),result.poses,
            (),0.,angular_audit=result.angular_audit,
        )
    if obstacles:
        raise FlightPhysicsError("torque-free collision sweep not yet certified for obstacles")
    highest=float("-inf")
    for patch in patches:
        for sx in (-1.,1.):
            for sz in (-1.,1.):
                corner=(patch.origin+patch.right*(sx*patch.width*.5)
                        +patch.forward*(sz*patch.length*.5))
                highest=max(highest,corner.y)
    endpoint_y=airborne.position.y+airborne.velocity.y*dt-.5*gravity*dt*dt
    radius=max(
        body.motion_radius_m,
        *(v.magnitude for v in policy.contact.wheels.positions),
    )
    if min(airborne.position.y,endpoint_y)-radius <= highest+1e-7:
        raise FlightPhysicsError("torque-free flight road clearance not globally certified")
    try:
        state,audit=advance_torque_free_rotation(
            AngularRigidBodyState(airborne.orientation,airborne.angular_velocity),
            opts,dt,
        )
    except RigidBodyIntegrationError as exc:
        raise FlightPhysicsError(f"torque-free angular integration blocked: {exc}") from exc
    end=VehicleState(
        airborne.position+airborne.velocity*dt+Vec3(0.,-.5*gravity*dt*dt,0.),
        airborne.velocity+Vec3(0.,-gravity*dt,0.),
        state.orientation,state.world_angular_velocity_radps,False,
    )
    return FlightStep(end,FlightState("airborne"),(airborne,end),(),0.,
                      angular_audit=audit)


def advance_ballistic_flight(
    airborne: VehicleState, policy: BallisticFlightPolicy,
    body: VehicleBodyGeometry, surfaces: tuple[PlanarSurfacePatch, ...],
    obstacles: tuple[OrientedBoxVolume, ...], *,
    dt: float, gravity_mps2: float,
    landing_dynamics: IndependentSuspensionDynamicsPolicy | None = None,
) -> FlightStep:
    """Advance exact gravity-only path, with certified early contact or rejection.

    LANDING result is an event-time state; caller must integrate remaining_dt_s
    using its registered grounded dynamics before claiming a full world step.
    """
    if (not isinstance(airborne, VehicleState) or airborne.grounded
            or not isinstance(policy, BallisticFlightPolicy)
            or not isinstance(body, VehicleBodyGeometry)):
        raise ValueError("airborne vehicle, body and flight policy required")
    if (isinstance(landing_dynamics, IndependentSuspensionDynamicsPolicy)
            and landing_dynamics.force_model != 'legacy_vertical_v1'):
        raise FlightPhysicsError('landing does not implement the road-relative force model')
    if (isinstance(dt, bool) or not isinstance(dt, (int, float))
            or not isfinite(dt) or dt <= 0.
            or isinstance(gravity_mps2, bool)
            or not isinstance(gravity_mps2, (int, float))
            or not isfinite(gravity_mps2) or gravity_mps2 <= 0.):
        raise ValueError("positive finite dt and gravity required")
    patches, polygons = _flat_atlas(surfaces, policy)
    if policy.torque_free is not None:
        return _certified_high_clearance_torque_free(
            airborne,policy,body,patches,obstacles,dt,gravity_mps2,
            landing_dynamics,
        )
    if policy.manifold is not None:
        from .manifold_flight import advance_manifold_flight
        return advance_manifold_flight(
            airborne,policy,body,(patches,polygons),obstacles,
            dt,gravity_mps2,landing_dynamics,
        )
    if airborne.angular_velocity.magnitude > _EPS and policy.angular_impulse is not None:
        return _rotating_impulse_flight(
            airborne,policy,body,patches,polygons,obstacles,dt,gravity_mps2,
            landing_dynamics,
        )
    if airborne.angular_velocity.magnitude > _EPS:
        # The existing OBB sweep interpolates orientation via shortest-arc
        # SLERP. Rotation > pi per sweep segment could alias a full turn and
        # incorrectly report no obstacle hit; use a much stricter bound.
        if airborne.angular_velocity.magnitude * dt / policy.collision_segments > .1:
            raise FlightPhysicsError("angular sweep resolution insufficient")
        _rotating_exit_certified(
            airborne, policy, body, patches, polygons, dt, gravity_mps2,
        )
        arc = _certify_obstacle_arc(
            airborne, dt, gravity_mps2, body, obstacles, policy.collision_segments,
        )
        return FlightStep(arc[-1], FlightState("airborne"), arc, (), 0.)
    # Reject zero-time descending road contact. Any later body intrusion
    # (including side entry into a bounded ramp) is certified by recursion.
    initial_footprint = body_xz_polygon(airborne, body)
    for patch, polygon in zip(patches, polygons, strict=True):
        closing_v = airborne.velocity.dot(patch.normal)
        if _intersection(initial_footprint, polygon):
            sep = min((p-patch.origin).dot(patch.normal)
                      for p in _body_corners(airborne, body))
            if sep < -_EPS or (sep <= _EPS and closing_v <= 0.):
                raise FlightPhysicsError("airborne chassis starts in road contact")
        for local in policy.contact.wheels.positions:
            p = airborne.position + airborne.orientation.rotate(local)
            if _within_shape(p.x, p.z, polygon):
                sep = (p-patch.origin).dot(patch.normal)
                if sep < -_EPS or (sep <= _EPS and closing_v <= 0.):
                    raise FlightPhysicsError("airborne wheel begins in descending road contact")
    events = _landable_events(airborne, dt, gravity_mps2, policy, body, patches, polygons)
    first = events[0] if events else None
    end_time = min(dt, first[0]) if first else dt
    try:
        certify_swept_chassis_clearance(
            lambda t: _flight_pose(airborne, t, gravity_mps2),
            0., end_time, body, (patches, polygons),
            max_subdivisions=12, gravity_mps2=gravity_mps2,
        )
    except SuspensionResolutionError as exc:
        raise FlightPhysicsError(f"uncertified airborne road/body sweep: {exc}") from exc
    poses = _certify_obstacle_arc(
        airborne, end_time, gravity_mps2, body, obstacles, policy.collision_segments,
    )
    if first is None:
        return FlightStep(poses[-1], FlightState("airborne"), poses, (), 0.)
    t, _, patch_id, collision_kind = first
    if collision_kind == "body":
        raise FlightPhysicsError("chassis contacts road before eligible wheel touchdown")
    impact = _flight_pose(airborne, t, gravity_mps2)
    impact_patch = next(p for p in patches if p.id == patch_id)
    down_speed = -impact.velocity.dot(impact_patch.normal) / impact_patch.normal.y
    if down_speed < 0. or down_speed > policy.max_touchdown_vertical_speed_mps + _EPS:
        raise FlightPhysicsError("unsafe touchdown vertical speed")
    up = impact.orientation.rotate(_UP)
    tilt = acos(max(-1., min(1., up.dot(_UP))))
    if tilt > policy.max_touchdown_tilt_rad + _EPS:
        raise FlightPhysicsError("touchdown tilt exceeds configured acceptance")
    # Research-only perfectly inelastic low-energy map: orientation is
    # preserved, all vertical and angular rates are zeroed, and the
    # reconstructed pitch/roll become persistent grounded state. This
    # is NOT a calibrated contact impulse/momentum-conservation model.
    landing_suspension = _stationary_landing_attitude(
        impact.orientation, impact.position.y,
    )
    landing = VehicleState(
        impact.position,
        Vec3(impact.velocity.x, 0., impact.velocity.z),
        impact.orientation, Vec3(0., 0., 0.), True,
    )
    support = certify_multisurface_support(landing, landing, body, policy.contact, surfaces)
    if not support.valid:
        raise FlightPhysicsError(f"touchdown road/wheel/body validation {support.status}: {support.reason}")
    # Ensure t wasn't preceded by an unmodeled road/body impact.
    return FlightStep(
        landing, FlightState("grounded"),
        (*poses[:-1], landing), (FlightTransition("landing", t, patch_id),),
        max(0., dt-t), down_speed, landing_suspension,
    )
