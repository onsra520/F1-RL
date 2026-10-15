"""Opt-in prospective 3-DOF independent suspension dynamics (research v1).

A heave/pitch/roll chassis integrates four independent unilateral spring-damper
contact forces by semi-implicit Euler, with deterministic fixed substeps. This
model deliberately does not solve lateral tyre forces, 6-DOF rigid-body
dynamics, airborne/landing or chassis mass/inertia calibration.

The longitudinal path/yaw is prescribed by the calling motion integrator.
Every integrated substep is checked by the separate conservative continuous
wheel/body road-coverage contract; failure rejects the rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin

from .body_collision import VehicleBodyGeometry, _slerp
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .multi_surface_support import (
    IndependentWheelContact, MultiSurfaceSuspensionPolicy,
    _atlas, _contacts, certify_multisurface_support,
)
from .track.patch import PlanarSurfacePatch
from .vehicle import VehicleState

_UP = Vec3(0., 1., 0.)
_X = Vec3(1., 0., 0.)
_Z = Vec3(0., 0., 1.)
_EPS = 1e-9


class SuspensionResolutionError(RuntimeError):
    """The opted-in physical candidate cannot be certified; never an ObstacleHit."""


@dataclass(frozen=True, slots=True)
class IndependentSuspensionDynamicsPolicy:
    contact: MultiSurfaceSuspensionPolicy
    mass_kg: float
    pitch_inertia_kg_m2: float
    roll_inertia_kg_m2: float
    spring_stiffness_npm: float
    damper_nspm: float
    angular_damping_nmsprad: float
    integration_substeps: int = 8
    max_abs_tilt_rad: float = .35
    angular_rate_mapping: str = "small_angle_v1"

    def __post_init__(self) -> None:
        if self.angular_rate_mapping not in ("small_angle_v1", "yaw_rx_rz_v1"):
            raise ValueError("unregistered suspension angular-rate mapping")
        if not isinstance(self.contact, MultiSurfaceSuspensionPolicy):
            raise ValueError("explicit independent wheel contact policy required")
        for key in ("mass_kg", "pitch_inertia_kg_m2", "roll_inertia_kg_m2",
                    "spring_stiffness_npm", "max_abs_tilt_rad"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite positive")
        for key in ("damper_nspm", "angular_damping_nmsprad"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v < 0:
                raise ValueError(f"{key} must be finite nonnegative")
        if type(self.integration_substeps) is not int or not 1 <= self.integration_substeps <= 128:
            raise ValueError("integration_substeps must be integer in [1,128]")
        if self.max_abs_tilt_rad >= .75:
            raise ValueError("max_abs_tilt_rad must be < 0.75 for small-angle model")


@dataclass(frozen=True, slots=True)
class SuspensionDynamicsState:
    height_m: float
    heave_velocity_mps: float = 0.
    pitch_x_rad: float = 0.
    pitch_rate_radps: float = 0.
    roll_z_rad: float = 0.
    roll_rate_radps: float = 0.

    def __post_init__(self) -> None:
        if any(not isfinite(getattr(self, name)) for name in (
            "height_m", "heave_velocity_mps", "pitch_x_rad", "pitch_rate_radps",
            "roll_z_rad", "roll_rate_radps",
        )):
            raise ValueError("suspension dynamics state must be finite")


@dataclass(frozen=True, slots=True)
class WheelForceSample:
    contact: IndependentWheelContact
    upward_force_n: float
    vertical_wheel_velocity_mps: float

    def __post_init__(self) -> None:
        if not isinstance(self.contact, IndependentWheelContact):
            raise ValueError("wheel force requires a contact")
        if (not isfinite(self.upward_force_n) or self.upward_force_n < 0.
                or not isfinite(self.vertical_wheel_velocity_mps)):
            raise ValueError("invalid wheel force telemetry")


@dataclass(frozen=True, slots=True)
class SuspensionDynamicsResult:
    vehicle: VehicleState
    dynamics: SuspensionDynamicsState
    poses: tuple[VehicleState, ...]
    force_samples: tuple[tuple[WheelForceSample, ...], ...]

    def __post_init__(self) -> None:
        if len(self.poses) != len(self.force_samples) + 1 or self.poses[-1] != self.vehicle:
            raise ValueError("inconsistent suspension trajectory")


def _horizontal_yaw(orientation: Quaternion) -> Quaternion:
    forward = orientation.rotate(_Z)
    horizontal = Vec3(forward.x, 0., forward.z)
    if horizontal.magnitude <= _EPS:
        raise SuspensionResolutionError("cannot infer yaw from vertical forward axis")
    horizontal = horizontal.normalized()
    return Quaternion.from_basis(_UP.cross(horizontal), _UP, horizontal)


def suspension_orientation(yaw: Quaternion, dynamics: SuspensionDynamicsState) -> Quaternion:
    """Local +X rotation is nose-down; local +Z rotation raises right side."""
    return (yaw
            * Quaternion.from_axis_angle(_X, dynamics.pitch_x_rad)
            * Quaternion.from_axis_angle(_Z, dynamics.roll_z_rad))


def angular_velocity_for_suspension(
    yaw: Quaternion, state: SuspensionDynamicsState, yaw_rate: float,
    mapping: str = "small_angle_v1",
) -> Vec3:
    if mapping == "yaw_rx_rz_v1":
        local = Vec3(state.pitch_rate_radps, -sin(state.pitch_x_rad)*state.roll_rate_radps,
                     cos(state.pitch_x_rad)*state.roll_rate_radps)
        return yaw.rotate(local) + Vec3(0., yaw_rate, 0.)
    if mapping != "small_angle_v1":
        raise ValueError("unregistered suspension angular-rate mapping")
    rotation_rate = yaw.rotate(Vec3(state.pitch_rate_radps, 0., state.roll_rate_radps))
    return Vec3(rotation_rate.x, yaw_rate, rotation_rate.z)


def _pose(x: float, z: float, yaw: Quaternion, state: SuspensionDynamicsState,
          horizontal_velocity: Vec3, yaw_rate: float,
          mapping: str = "small_angle_v1") -> VehicleState:
    angle = suspension_orientation(yaw, state)
    return VehicleState(
        Vec3(x, state.height_m, z),
        Vec3(horizontal_velocity.x, state.heave_velocity_mps, horizontal_velocity.z),
        angle, angular_velocity_for_suspension(yaw, state, yaw_rate, mapping), True,
    )


def integrate_independent_suspension(
    previous: VehicleState,
    path_candidate: VehicleState,
    initial: SuspensionDynamicsState,
    policy: IndependentSuspensionDynamicsPolicy,
    body: VehicleBodyGeometry,
    surfaces: tuple[PlanarSurfacePatch, ...],
    *,
    dt: float,
    gravity_mps2: float,
) -> SuspensionDynamicsResult:
    """Integrate unilateral 4-wheel forces along the supplied XZ/yaw path.

    This solver owns chassis height, pitch and roll, but not XZ acceleration,
    steering or yaw. The *discrete polygonal* trajectory is checked segment
    by segment; no claim of continuous integration of real-world force ODEs.
    """
    if (not isinstance(previous, VehicleState) or not isinstance(path_candidate, VehicleState)
            or not isinstance(initial, SuspensionDynamicsState)
            or not isinstance(policy, IndependentSuspensionDynamicsPolicy)
            or not isinstance(body, VehicleBodyGeometry)):
        raise ValueError("invalid suspension dynamics inputs")
    if (isinstance(dt, bool) or not isinstance(dt, (int, float)) or
            not isfinite(dt) or dt <= 0.):
        raise ValueError("dt must be finite positive")
    if (isinstance(gravity_mps2, bool) or not isinstance(gravity_mps2, (int, float))
            or not isfinite(gravity_mps2) or gravity_mps2 <= 0.):
        raise ValueError("gravity_mps2 must be finite positive")
    if not previous.grounded or not path_candidate.grounded:
        raise SuspensionResolutionError("airborne and landing dynamics not registered")
    atlas = _atlas(surfaces, policy.contact.seam_height_tolerance_m)
    if atlas is None:
        raise SuspensionResolutionError("non-2.5D or discontinuous multi-surface road atlas")
    if abs(previous.position.y - initial.height_m) > 1e-7:
        raise SuspensionResolutionError("stored chassis height disagrees with vehicle pose")

    yaw0, yaw1 = _horizontal_yaw(previous.orientation), _horizontal_yaw(path_candidate.orientation)
    expected = suspension_orientation(yaw0, initial)
    dot = abs(sum(a*b for a,b in zip(
        (expected.w,expected.x,expected.y,expected.z),
        (previous.orientation.w,previous.orientation.x,previous.orientation.y,previous.orientation.z),
        strict=True,
    )))
    if 1. - dot > 1e-9:
        raise SuspensionResolutionError("stored chassis attitude disagrees with vehicle orientation")
    if (abs(initial.pitch_x_rad) > policy.max_abs_tilt_rad or
            abs(initial.roll_z_rad) > policy.max_abs_tilt_rad):
        raise SuspensionResolutionError("initial chassis tilt exceeds small-angle contract")

    count = policy.integration_substeps
    h = dt / count
    pose = previous
    dynamic = initial
    route = path_candidate.position - previous.position
    poses = [pose]
    history: list[tuple[WheelForceSample, ...]] = []
    for index in range(count):
        contact = _contacts(pose, policy.contact, atlas)
        if contact is None:
            raise SuspensionResolutionError(f"wheel support/travel missing at substep {index}")
        yaw = _slerp(yaw0, yaw1, (index + 1) / count)
        force_samples = []
        total_force = 0.
        pitch_torque = -policy.angular_damping_nmsprad * dynamic.pitch_rate_radps
        roll_torque = -policy.angular_damping_nmsprad * dynamic.roll_rate_radps
        for local, ground in zip(policy.contact.wheels.positions, contact, strict=True):
            # Linearized wheel vertical velocity in chassis local pitch/roll
            # coordinates; consistent with the registered small-angle bound.
            wheel_vel = (dynamic.heave_velocity_mps
                         - dynamic.pitch_rate_radps * local.z
                         + dynamic.roll_rate_radps * local.x)
            preload = policy.mass_kg * gravity_mps2 / 4.
            load = max(0., preload + policy.spring_stiffness_npm * ground.deflection_m
                       - policy.damper_nspm * wheel_vel)
            force_samples.append(WheelForceSample(ground, load, wheel_vel))
            total_force += load
            pitch_torque -= local.z * load
            roll_torque += local.x * load
        new_heave_v = dynamic.heave_velocity_mps + (
            total_force / policy.mass_kg - gravity_mps2) * h
        new_pitch_v = dynamic.pitch_rate_radps + (
            pitch_torque / policy.pitch_inertia_kg_m2) * h
        new_roll_v = dynamic.roll_rate_radps + (
            roll_torque / policy.roll_inertia_kg_m2) * h
        dynamic = SuspensionDynamicsState(
            dynamic.height_m + new_heave_v * h,
            new_heave_v,
            dynamic.pitch_x_rad + new_pitch_v * h,
            new_pitch_v,
            dynamic.roll_z_rad + new_roll_v * h,
            new_roll_v,
        )
        if (abs(dynamic.pitch_x_rad) > policy.max_abs_tilt_rad or
                abs(dynamic.roll_z_rad) > policy.max_abs_tilt_rad):
            raise SuspensionResolutionError(f"tilt bound exceeded at substep {index}")
        t = (index + 1) / count
        next_pose = _pose(
            previous.position.x + route.x * t,
            previous.position.z + route.z * t,
            yaw, dynamic, path_candidate.velocity, path_candidate.angular_velocity.y,
            policy.angular_rate_mapping,
        )
        report = certify_multisurface_support(pose, next_pose, body, policy.contact, surfaces)
        if not report.valid:
            raise SuspensionResolutionError(
                f"uncertified suspension substep {index}: {report.status}: {report.reason}"
            )
        history.append(tuple(force_samples))
        poses.append(next_pose)
        pose = next_pose
    return SuspensionDynamicsResult(pose, dynamic, tuple(poses), tuple(history))
