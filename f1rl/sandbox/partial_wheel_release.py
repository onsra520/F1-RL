"""WP7 research: independently loaded wheels and automatic platform-edge release.

This is an opt-in, flat 2.5D platform model. A missing road projection makes
that wheel *unloaded*; a wheel that remains above the road but exceeds travel
is NOT silently classified as leaving the road. Transition time is localized
within a deterministic bracket; it is not a continuous-time force ODE proof.

The dynamics remain the prospective small-angle 3-DOF spring model, with
legacy-prescribed XZ/yaw. Do not use this as an accepted 6-DOF jump model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, atan2, hypot

from .body_collision import VehicleBodyGeometry, _slerp
from .geometry.vector import Vec3
from .independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
    SuspensionResolutionError, _horizontal_yaw, _pose, suspension_orientation,
)
from .multi_surface_support import (
    IndependentWheelContact, _atlas, _within_shape, _height, _body_corners,
    _intersection, _inflate, body_xz_polygon,
)
from .swept_surface_clearance import certify_swept_chassis_clearance
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState

_EPS = 1e-9
_WHEELS = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True, slots=True)
class AutomaticTakeoffPolicy:
    """Engineering constraints; values are NOT science-accepted thresholds."""
    max_substep_xz_m: float
    release_time_tolerance_s: float = 1e-6
    release_bisections: int = 24
    max_patch_tilt_rad: float = 0.  # opt-in: retain legacy flat-only default

    def __post_init__(self) -> None:
        for k in ("max_substep_xz_m", "release_time_tolerance_s"):
            value = getattr(self, k)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0.:
                raise ValueError(f"{k} must be finite positive")
        if (type(self.max_patch_tilt_rad) not in (int,float)
                or not isfinite(self.max_patch_tilt_rad)
                or not 0. <= self.max_patch_tilt_rad <= .35):
            raise ValueError("max_patch_tilt_rad must be finite within [0,0.35]")
        if type(self.release_bisections) is not int or not 4 <= self.release_bisections <= 48:
            raise ValueError("release_bisections must be an integer in [4,48]")


@dataclass(frozen=True, slots=True)
class PartialWheelFrame:
    time_s: float
    contacts: tuple[IndependentWheelContact | None, ...]
    forces_n: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (len(self.contacts) != 4 or len(self.forces_n) != 4
                or not isfinite(self.time_s) or self.time_s < 0.
                or any(not isfinite(v) or v < 0. for v in self.forces_n)):
            raise ValueError("invalid four-wheel contact telemetry")

    @property
    def supported_wheels(self) -> int:
        return sum(c is not None for c in self.contacts)


@dataclass(frozen=True, slots=True)
class GroundSupportStep:
    vehicle: VehicleState
    dynamics: SuspensionDynamicsState
    poses: tuple[VehicleState, ...]
    frames: tuple[PartialWheelFrame, ...]
    takeoff_at_s: float | None
    remaining_dt_s: float

    def __post_init__(self) -> None:
        if not self.poses or self.poses[-1] != self.vehicle:
            raise ValueError("last recorded pose must be final candidate")
        if not isfinite(self.remaining_dt_s) or self.remaining_dt_s < 0.:
            raise ValueError("invalid residual frame duration")


def _flat_atlas(surfaces, tol, max_patch_tilt_rad=0.):
    atlas = _atlas(surfaces, tol)
    if atlas is None:
        raise SuspensionResolutionError("unsupported automatic takeoff road atlas")
    patches, polygons = atlas
    if any(atan2(hypot(p.normal.x, p.normal.z), p.normal.y)
           > max_patch_tilt_rad + _EPS for p in patches):
        raise SuspensionResolutionError("road slope exceeds registered release policy")
    return patches, polygons


def sample_partial_wheels(state: VehicleState, contact_policy, atlas):
    """Return four independent contacts, with None only outside every patch."""
    patches, polys = atlas
    result = []
    for name, local in zip(_WHEELS, contact_policy.wheels.positions, strict=True):
        point = state.position + state.orientation.rotate(local)
        candidates = []
        for patch, poly in zip(patches, polys, strict=True):
            if not _within_shape(point.x, point.z, poly):
                continue
            ground_y = _height(patch, point.x, point.z)
            d = ground_y - point.y
            if d > contact_policy.max_bump_m + _EPS:
                raise SuspensionResolutionError("wheel penetrates road beyond bump travel")
            if d < -contact_policy.max_droop_m - _EPS:
                raise SuspensionResolutionError("wheel above road exceeds droop; no edge release inferred")
            candidates.append((abs(d), patch.id, ground_y, d))
        if candidates:
            _, chosen, height, d = min(candidates)
            result.append(IndependentWheelContact(name, chosen, height, d))
        else:
            result.append(None)
    return tuple(result)


def _pose_between(a: VehicleState, b: VehicleState, t: float):
    return VehicleState(
        a.position + (b.position - a.position)*t,
        a.velocity + (b.velocity-a.velocity)*t,
        _slerp(a.orientation, b.orientation, t),
        a.angular_velocity + (b.angular_velocity-a.angular_velocity)*t,
        True,
    )


def _body_road_clear(a, b, body, atlas):
    """Prove entire swept 3D chassis remains above relevant planar patches."""
    certify_swept_chassis_clearance(
        lambda t: _pose_between(a, b, t),
        0., 1., body, atlas, max_subdivisions=12,
    )


def _step_dynamics(state, initial, contacts, physics, gravity, h):
    total = 0.
    pitch_torque = -physics.angular_damping_nmsprad*initial.pitch_rate_radps
    roll_torque = -physics.angular_damping_nmsprad*initial.roll_rate_radps
    forces = []
    for local, contact in zip(physics.contact.wheels.positions, contacts, strict=True):
        if contact is None:
            forces.append(0.)
            continue
        wheel_v = (initial.heave_velocity_mps
                   - initial.pitch_rate_radps*local.z
                   + initial.roll_rate_radps*local.x)
        preload = physics.mass_kg*gravity/4.
        load = max(0., preload+physics.spring_stiffness_npm*contact.deflection_m
                   - physics.damper_nspm*wheel_v)
        forces.append(load)
        total += load
        pitch_torque -= local.z*load
        roll_torque += local.x*load
    vy = initial.heave_velocity_mps+(total/physics.mass_kg-gravity)*h
    vp = initial.pitch_rate_radps+(pitch_torque/physics.pitch_inertia_kg_m2)*h
    vr = initial.roll_rate_radps+(roll_torque/physics.roll_inertia_kg_m2)*h
    candidate = SuspensionDynamicsState(
        initial.height_m+vy*h, vy,
        initial.pitch_x_rad+vp*h, vp,
        initial.roll_z_rad+vr*h, vr,
    )
    if abs(candidate.pitch_x_rad) > physics.max_abs_tilt_rad or abs(candidate.roll_z_rad) > physics.max_abs_tilt_rad:
        raise SuspensionResolutionError("partial-wheel dynamic tilt exceeds small-angle contract")
    return candidate, tuple(forces)


def integrate_partial_support(
    previous: VehicleState,
    path_candidate: VehicleState,
    initial: SuspensionDynamicsState,
    physics: IndependentSuspensionDynamicsPolicy,
    body: VehicleBodyGeometry,
    surfaces: tuple[PlanarSurfacePatch, ...],
    release_policy: AutomaticTakeoffPolicy,
    *, dt: float, gravity_mps2: float,
) -> GroundSupportStep:
    if (not isinstance(previous, VehicleState) or not isinstance(path_candidate, VehicleState)
            or not isinstance(initial, SuspensionDynamicsState)
            or not isinstance(physics, IndependentSuspensionDynamicsPolicy)
            or not isinstance(body, VehicleBodyGeometry)
            or not isinstance(release_policy, AutomaticTakeoffPolicy)):
        raise ValueError("invalid partial-wheel integration inputs")
    if not previous.grounded:
        raise SuspensionResolutionError("partial-wheel solver requires grounded start")
    if (type(dt) not in (float,int) or not isfinite(dt) or dt <= 0.
            or type(gravity_mps2) not in (float,int) or not isfinite(gravity_mps2) or gravity_mps2 <= 0.):
        raise ValueError("positive finite dt and gravity required")
    if abs(previous.position.y-initial.height_m) > 1e-7:
        raise SuspensionResolutionError("stored suspension height mismatch")
    yaw0, yaw1 = _horizontal_yaw(previous.orientation), _horizontal_yaw(path_candidate.orientation)
    expected = suspension_orientation(yaw0, initial)
    q = previous.orientation
    if 1.-abs(expected.w*q.w+expected.x*q.x+expected.y*q.y+expected.z*q.z) > 1e-9:
        raise SuspensionResolutionError("stored suspension attitude mismatch")
    atlas = _flat_atlas(surfaces, physics.contact.seam_height_tolerance_m,
                        release_policy.max_patch_tilt_rad)
    count = physics.integration_substeps
    h = dt/count
    route = path_candidate.position-previous.position
    if (Vec3(route.x,0.,route.z).magnitude/count > release_policy.max_substep_xz_m):
        raise SuspensionResolutionError("substep XZ movement exceeds certified release resolution")
    dynamic, pose = initial, previous
    frames = []
    poses = [previous]
    prior_mask = None
    for i in range(count):
        contacts = sample_partial_wheels(pose, physics.contact, atlas)
        mask = tuple(c is not None for c in contacts)
        if not any(mask):
            raise SuspensionResolutionError("grounded pose has zero supported wheels")
        if prior_mask is not None and any(m and not o for o,m in zip(prior_mask, mask, strict=True)):
            raise SuspensionResolutionError("unregistered wheel recontact during edge release")
        new_dynamic, forces = _step_dynamics(
            pose, dynamic, contacts, physics, gravity_mps2, h,
        )
        frames.append(PartialWheelFrame(i*h, contacts, forces))
        t = (i+1)/count
        yaw = _slerp(yaw0,yaw1,t)
        proposed = _pose(
            previous.position.x+route.x*t, previous.position.z+route.z*t,
            yaw, new_dynamic, path_candidate.velocity, path_candidate.angular_velocity.y,
            physics.angular_rate_mapping,
        )
        _body_road_clear(pose, proposed, body, atlas)
        final_contacts = sample_partial_wheels(proposed, physics.contact, atlas)
        final_mask = tuple(c is not None for c in final_contacts)
        # This research increment only allows monotonically disappearing
        # wheel support inside each fixed substep; recontact needs impulses.
        if any(end and not start for start,end in zip(mask, final_mask, strict=True)):
            raise SuspensionResolutionError("unsupported newly contacting wheel during step")
        if not any(final_mask):
            lo, hi = 0., 1.
            for _ in range(release_policy.release_bisections):
                if h*(hi-lo) <= release_policy.release_time_tolerance_s:
                    break
                mid = (lo+hi)/2.
                guess = _pose_between(pose, proposed, mid)
                if any(sample_partial_wheels(guess, physics.contact, atlas)):
                    lo = mid
                else:
                    hi = mid
            launch = _pose_between(pose, proposed, hi)
            if any(sample_partial_wheels(launch, physics.contact, atlas)):
                raise SuspensionResolutionError("takeoff edge localization not resolved")
            _body_road_clear(pose, launch, body, atlas)
            departure = (i+hi)*h
            # The discretized force update is linearly interpolated to the
            # located release; its force ODE error is not certified.
            dynamic = SuspensionDynamicsState(
                launch.position.y,
                dynamic.heave_velocity_mps + (new_dynamic.heave_velocity_mps-dynamic.heave_velocity_mps)*hi,
                dynamic.pitch_x_rad + (new_dynamic.pitch_x_rad-dynamic.pitch_x_rad)*hi,
                dynamic.pitch_rate_radps + (new_dynamic.pitch_rate_radps-dynamic.pitch_rate_radps)*hi,
                dynamic.roll_z_rad + (new_dynamic.roll_z_rad-dynamic.roll_z_rad)*hi,
                dynamic.roll_rate_radps + (new_dynamic.roll_rate_radps-dynamic.roll_rate_radps)*hi,
            )
            air = VehicleState(launch.position,launch.velocity,launch.orientation,
                               launch.angular_velocity,False)
            poses.append(air)
            return GroundSupportStep(
                air, dynamic, tuple(poses), tuple(frames),
                departure, max(0., dt-departure),
            )
        poses.append(proposed)
        prior_mask = mask
        dynamic, pose = new_dynamic, proposed
    return GroundSupportStep(pose, dynamic, tuple(poses), tuple(frames), None, 0.)
