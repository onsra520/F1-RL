"""Conservative CCD for the *defined* piecewise-RK4 torque-free flight path.

Euler/RK4 produces orientation and spin at fixed interval anchors. The
simulation's continuous in-between pose is shortest-arc SLERP between each
pair of anchors plus the exact gravity-only parabola for centre position.
The CCD certificate covers THAT prescribed numerical trajectory, including
the ballistic chord sagitta; it does not claim validated error bounds against
the analytic torque-free Euler ODE or authorize an actual touchdown.

Both rotating OBB-vs-OBB and wheel/chassis-vs-road clearance are checked.
Uncertain leaves and any witnessed contact reject the flight candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite

from .body_collision import VehicleBodyGeometry, _slerp, sweep_body_against_obstacle
from .geometry.vector import Vec3
from .independent_suspension_dynamics import SuspensionResolutionError
from .multi_surface_support import _intersection, _square, _within_shape
from .swept_surface_clearance import certify_swept_chassis_clearance
from .torque_free_rotation import (
    AngularRigidBodyAudit, AngularRigidBodyState, RigidBodyIntegrationError,
    TorqueFreeRotationPolicy, advance_torque_free_rotation,
)
from .track.volume import OrientedBoxVolume
from .vehicle import VehicleState

_EPS=1e-9
_VERTICAL=Vec3(0.,1.,0.)


class TorqueFreeSweepResolutionError(RuntimeError):
    """Contact or insufficient numerical clearance certification."""


@dataclass(frozen=True,slots=True)
class TorqueFreeSweepPolicy:
    slices: int=32
    max_subdivisions: int=12
    max_slice_rotation_rad: float=.12
    clearance_epsilon_m: float=1e-7

    def __post_init__(self):
        if type(self.slices) is not int or not 4<=self.slices<=256:
            raise ValueError("slices must be integer in [4,256]")
        if type(self.max_subdivisions) is not int or not 0<=self.max_subdivisions<=20:
            raise ValueError("max_subdivisions must be integer in [0,20]")
        for name in ("max_slice_rotation_rad","clearance_epsilon_m"):
            x=getattr(self,name)
            if type(x) not in (int,float) or not isfinite(x) or x<=0.:
                raise ValueError(name+" must be finite positive")
        if self.max_slice_rotation_rad>.3 or self.clearance_epsilon_m>.01:
            raise ValueError("collision certificate tolerance outside registered bound")


@dataclass(frozen=True,slots=True)
class CertifiedTorqueFreeFlight:
    poses: tuple[VehicleState,...]
    angular_audit: AngularRigidBodyAudit
    obstacle_checks: int
    road_checks: int
    wheel_checks: int

    @property
    def vehicle(self):
        return self.poses[-1]


def _pose_between(a,b,start_s,segment_s,gravity,t):
    alpha=t/segment_s
    # Gravity centre curve is exact, not the linear chord used by an OBB sweep.
    position=a.position+a.velocity*t+Vec3(0.,-.5*gravity*t*t,0.)
    velocity=a.velocity+Vec3(0.,-gravity*t,0.)
    return VehicleState(
        position,velocity,_slerp(a.orientation,b.orientation,alpha),
        a.angular_velocity+(b.angular_velocity-a.angular_velocity)*alpha,
        False,
    )


def _wheel_clearance(pose_at,segment_s,gravity,wheels,atlas,policy):
    patches,polygons=atlas
    count=0
    for local in wheels.positions:
        arm=local.magnitude
        def sample(t):
            nonlocal count
            count+=1
            vehicle=pose_at(t)
            return vehicle.position+vehicle.orientation.rotate(local)
        def visit(lo,hi,depth):
            center=(lo+hi)*.5
            mid=sample(center)
            a=pose_at(lo)
            b=pose_at(hi)
            dot=abs(a.orientation.w*b.orientation.w
                    +a.orientation.x*b.orientation.x
                    +a.orientation.y*b.orientation.y
                    +a.orientation.z*b.orientation.z)
            rotation=2.*acos(max(0.,min(1.,dot)))
            delta=(b.position-a.position).magnitude
            # For piecewise SLERP + parabolic centre translation:
            # all intermediate wheel points are within this radius of
            # midpoint. The final term encloses gravity's chord sagitta.
            bound=.5*(delta+rotation*arm)+gravity*(hi-lo)**2/8.
            envelope=_square(mid.x,mid.z,bound+_EPS)
            maybe=[]
            for patch,poly in zip(patches,polygons,strict=True):
                if not _intersection(envelope,poly):
                    continue
                gap=(mid-patch.origin).dot(patch.normal)
                if gap<=policy.clearance_epsilon_m and _within_shape(mid.x,mid.z,poly):
                    raise TorqueFreeSweepResolutionError("wheel/road contact or penetration witnessed")
                if gap <= bound+policy.clearance_epsilon_m:
                    maybe.append(patch.id)
            if not maybe:
                return
            if depth==0 or center==lo or center==hi:
                raise TorqueFreeSweepResolutionError(
                    "wheel/road sweep unresolved within registered subdivision budget"
                )
            visit(lo,center,depth-1)
            visit(center,hi,depth-1)
        for endpoint in (0.,segment_s):
            p=sample(endpoint)
            for patch,poly in zip(patches,polygons,strict=True):
                if _within_shape(p.x,p.z,poly):
                    if (p-patch.origin).dot(patch.normal)<=policy.clearance_epsilon_m:
                        raise TorqueFreeSweepResolutionError(
                            "wheel starts/ends at road contact or penetration"
                        )
        visit(0.,segment_s,policy.max_subdivisions)
    return count


def certify_torque_free_flight_sweep(
    initial: VehicleState,
    angular: TorqueFreeRotationPolicy,
    body: VehicleBodyGeometry,
    wheels,
    atlas,
    obstacles: tuple[OrientedBoxVolume,...],
    *,
    duration_s: float,
    gravity_mps2: float,
    sweep: TorqueFreeSweepPolicy,
)->CertifiedTorqueFreeFlight:
    """Certify no road/body/wheel or obstacle contact over RK4 anchor splines.

    Segment OBB sweep uses exact SLERP rotational interpolation and expands
    each obstacle OBB along its own local axes to contain the world-vertical
    parabola sagitta. Road sweep checks entire OBB, not just wheel samples.
    """
    from .ramp_support import WheelContactPoints
    if (not isinstance(initial,VehicleState) or initial.grounded
            or not isinstance(angular,TorqueFreeRotationPolicy)
            or not isinstance(body,VehicleBodyGeometry)
            or not isinstance(wheels,WheelContactPoints)
            or not isinstance(sweep,TorqueFreeSweepPolicy)
            or not isinstance(obstacles,tuple)
            or any(not isinstance(o,OrientedBoxVolume) for o in obstacles)):
        raise ValueError("registered airborne, body, wheel, sweep and obstacle input required")
    if (type(duration_s) not in (int,float) or not isfinite(duration_s) or duration_s<=0.
            or type(gravity_mps2) not in (int,float) or not isfinite(gravity_mps2)
            or gravity_mps2<=0.):
        raise ValueError("finite positive duration and gravity required")
    patches,polygons=atlas
    if len(patches)!=len(polygons):
        raise ValueError("invalid registered road atlas")
    h=duration_s/sweep.slices
    poses=[initial]
    angular_state=AngularRigidBodyState(
        initial.orientation,initial.angular_velocity,
    )
    first_audit=None
    last_audit=None
    for i in range(sweep.slices):
        try:
            angular_state,audit=advance_torque_free_rotation(angular_state,angular,h)
        except RigidBodyIntegrationError as exc:
            raise TorqueFreeSweepResolutionError(
                f"torque-free RK4 integration uncertified: {exc}"
            ) from exc
        if first_audit is None:
            first_audit=audit
        last_audit=audit
        t=(i+1)*h
        pose=VehicleState(
            initial.position+initial.velocity*t+Vec3(0.,-.5*gravity_mps2*t*t,0.),
            initial.velocity+Vec3(0.,-gravity_mps2*t,0.),
            angular_state.orientation,angular_state.world_angular_velocity_radps,False,
        )
        before=poses[-1]
        dot=abs(before.orientation.w*pose.orientation.w
                +before.orientation.x*pose.orientation.x
                +before.orientation.y*pose.orientation.y
                +before.orientation.z*pose.orientation.z)
        theta=2.*acos(min(1.,max(0.,dot)))
        if theta>sweep.max_slice_rotation_rad:
            raise TorqueFreeSweepResolutionError(
                "RK4 anchor rotation too large to certify non-aliased sweep"
            )
        poses.append(pose)
    # A sequence of individually passing integration checks can still
    # accumulate unacceptable drift. Compare the first/last invariants.
    global_audit=AngularRigidBodyAudit(
        first_audit.initial_energy_j,last_audit.final_energy_j,
        first_audit.initial_world_angular_momentum,
        last_audit.final_world_angular_momentum,
        angular.integration_steps*sweep.slices,
    )
    if (abs(global_audit.final_energy_j-global_audit.initial_energy_j)
            >angular.energy_relative_tolerance*max(1.,global_audit.initial_energy_j)):
        raise TorqueFreeSweepResolutionError("global RK4 energy drift exceeds registered bound")
    if ((global_audit.final_world_angular_momentum
         -global_audit.initial_world_angular_momentum).magnitude
            >angular.momentum_relative_tolerance*max(
                1.,global_audit.initial_world_angular_momentum.magnitude)):
        raise TorqueFreeSweepResolutionError("global RK4 angular momentum drift exceeds bound")
    road_count=0
    wheel_count=0
    obstacle_count=0
    for i,(start,end) in enumerate(zip(poses,poses[1:])):
        # The simulation defines orientation *within* each slice by SLERP
        # between these two RK4 anchors; CCD therefore never estimates
        # the curvature of an undocumented interpolation.
        pose_at=lambda t,a=start,b=end: _pose_between(
            a,b,0.,h,gravity_mps2,t,
        )
        try:
            road_count+=certify_swept_chassis_clearance(
                pose_at,0.,h,body,atlas,
                max_subdivisions=sweep.max_subdivisions,
                gravity_mps2=gravity_mps2,
            )
        except SuspensionResolutionError as exc:
            raise TorqueFreeSweepResolutionError(
                f"chassis/road CCD rejected slice {i}: {exc}"
            ) from exc
        wheel_count+=_wheel_clearance(
            pose_at,h,gravity_mps2,wheels,atlas,sweep,
        )
        sagitta=gravity_mps2*h*h/8.
        for j,obstacle in enumerate(obstacles):
            q=obstacle.orientation
            ax=q.rotate(Vec3(1.,0.,0.))
            ay=q.rotate(Vec3(0.,1.,0.))
            az=q.rotate(Vec3(0.,0.,1.))
            expanded=OrientedBoxVolume(
                obstacle.center,
                Vec3(obstacle.half_extents.x+sagitta*abs(ax.y),
                     obstacle.half_extents.y+sagitta*abs(ay.y),
                     obstacle.half_extents.z+sagitta*abs(az.y)),
                q,
            )
            verdict=sweep_body_against_obstacle(
                start,end,body,expanded,
                max_subdivisions=sweep.max_subdivisions,
            )
            obstacle_count+=verdict.checks
            if verdict.blocked:
                raise TorqueFreeSweepResolutionError(
                    f"body/obstacle CCD rejected slice {i}, obstacle {j}: "
                    f"{'contact' if verdict.contact else 'unresolved'}"
                )
    return CertifiedTorqueFreeFlight(
        tuple(poses),global_audit,obstacle_count,road_count,wheel_count,
    )
