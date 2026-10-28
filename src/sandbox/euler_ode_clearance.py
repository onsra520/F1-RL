"""Analytic *torque-free Euler ODE* swept-clearance enclosure (WP7 research).

Unlike the piecewise RK4/SLERP simulation CCD, this certificate does not
assume RK4 approximates the true attitude. Under an exactly torque-free
rigid body with a fixed positive-definite body inertia, kinetic energy is
constant. Therefore all exact orientations satisfy

    ||omega(t)|| <= sqrt(2*E(0) / min(Ix,Iy,Iz))

and each material point can move from its EXACT initial orientation by at
most radius * min(omega_bound*t, 2). This crude but rigorous global envelope
is combined with the exact ballistic centre parabola and convex OBB/road
separation checks. Unknown proximity is REJECTED, not declared collision.

This proves clearance only, not exact collision time, contact force, tire
dynamics, numerical integration accuracy, or scientific calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .body_collision import VehicleBodyGeometry, obb_separation_lower_bound
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .multi_surface_support import _body_corners, _intersection, _square
from .road_footprint import _inflate, body_xz_polygon
from .torque_free_rotation import TorqueFreeRotationPolicy
from .vehicle import VehicleState

_DOWN=Vec3(0.,-1.,0.)
_EPS=1e-10


@dataclass(frozen=True,slots=True)
class EulerODEEnclosurePolicy:
    max_subdivisions: int=12
    min_clearance_m: float=1e-7

    def __post_init__(self):
        if type(self.max_subdivisions) is not int or not 0<=self.max_subdivisions<=20:
            raise ValueError("max_subdivisions must be in [0,20]")
        if (type(self.min_clearance_m) not in (int,float)
                or not isfinite(self.min_clearance_m)
                or not 0.<self.min_clearance_m<=.01):
            raise ValueError("min_clearance_m must be finite in (0,0.01]")


@dataclass(frozen=True,slots=True)
class EulerODEClearanceVerdict:
    status: str
    checks: int
    angular_speed_bound_radps: float
    first_uncertain_window_s: tuple[float,float] | None = None

    def __post_init__(self):
        if self.status not in ("clear","uncertain"):
            raise ValueError("invalid Euler ODE enclosure verdict")

    @property
    def valid(self):
        return self.status=="clear"


def _centre(initial,t,gravity):
    return (initial.position+initial.velocity*t
            +_DOWN*(.5*gravity*t*t))


def certify_torque_free_ode_clearance(
    initial: VehicleState,
    angular: TorqueFreeRotationPolicy,
    body: VehicleBodyGeometry,
    wheels,
    atlas: tuple,
    obstacles: tuple,
    *,
    duration_s: float,
    gravity_mps2: float,
    policy: EulerODEEnclosurePolicy,
)->EulerODEClearanceVerdict:
    """Conservatively enclose every exact torque-free pose for t in [0,dt].

    Clear means analytically separated for all time; unresolved means a
    bounded earliest interval contains *possible*, NOT proven, collision.
    A witness at a time using numeric RK4 alone is never accepted as an
    analytic Euler-ODE collision assertion.
    """
    from .ramp_support import WheelContactPoints
    from .track.volume import OrientedBoxVolume

    if (not isinstance(initial,VehicleState) or initial.grounded
            or not isinstance(angular,TorqueFreeRotationPolicy)
            or not isinstance(body,VehicleBodyGeometry)
            or not isinstance(wheels,WheelContactPoints)
            or not isinstance(policy,EulerODEEnclosurePolicy)
            or not isinstance(atlas,tuple) or len(atlas)!=2
            or not isinstance(obstacles,tuple)
            or any(not isinstance(o,OrientedBoxVolume) for o in obstacles)):
        raise ValueError("explicit airborne/body/wheel/inertia/atlas/obstacle policy required")
    if (type(duration_s) not in (int,float) or not isfinite(duration_s)
            or duration_s<=0.
            or type(gravity_mps2) not in (int,float) or not isfinite(gravity_mps2)
            or gravity_mps2<=0.):
        raise ValueError("finite positive time and gravity required")
    patches,polygons=atlas
    if len(patches)!=len(polygons):
        raise ValueError("invalid 2.5D atlas")

    q=initial.orientation
    inv=Quaternion(q.w,-q.x,-q.y,-q.z)
    spin=inv.rotate(initial.angular_velocity)
    Ix,Iy,Iz=(angular.pitch_inertia_kg_m2,angular.yaw_inertia_kg_m2,
              angular.roll_inertia_kg_m2)
    kinetic=.5*(Ix*spin.x**2+Iy*spin.y**2+Iz*spin.z**2)
    speed_bound=sqrt(2.*kinetic/min(Ix,Iy,Iz))
    if speed_bound>angular.max_angular_speed_radps:
        raise ValueError("initial angular speed energy bound exceeds registered domain")

    visits=0
    def visit(lo,hi,depth):
        nonlocal visits
        visits+=1
        mid=.5*(lo+hi)
        span=hi-lo
        centre=_centre(initial,mid,gravity_mps2)
        centre_velocity=initial.velocity+_DOWN*(gravity_mps2*mid)
        translation=.5*centre_velocity.magnitude*span+gravity_mps2*span*span/8.
        # Distance of a rigid point from its initial orientation under ANY
        # analytic Euler-ODE attitude; no RK4 anchors are trusted here.
        max_rot_angle=speed_bound*hi
        radius=translation+min(max_rot_angle,2.)*max(
            body.motion_radius_m,*(w.magnitude for w in wheels.positions),
        )
        margin=radius+policy.min_clearance_m
        sample=VehicleState(centre,initial.velocity,q,initial.angular_velocity,False)
        for obstacle in obstacles:
            gap=obb_separation_lower_bound(
                body.box_at(sample.position,sample.orientation),obstacle,
            )
            if gap<=margin:
                break
        else:
            # Static initial orientation OBB footprint expanded by the
            # entire analytic position+orientation uncertainty ball.
            shape=_inflate(body_xz_polygon(sample,body),margin)
            corners=_body_corners(sample,body)
            for patch,poly in zip(patches,polygons,strict=True):
                if not _intersection(shape,poly):
                    continue
                if min((v-patch.origin).dot(patch.normal) for v in corners)<=margin:
                    break
            else:
                for local in wheels.positions:
                    pos=sample.position+q.rotate(local)
                    footprint=_square(pos.x,pos.z,margin)
                    for patch,poly in zip(patches,polygons,strict=True):
                        if _intersection(footprint,poly):
                            if (pos-patch.origin).dot(patch.normal)<=margin:
                                break
                    else:
                        continue
                    break
                else:
                    return None
        if depth==0 or mid==lo or mid==hi:
            return (lo,hi)
        left=visit(lo,mid,depth-1)
        if left is not None:
            return left
        return visit(mid,hi,depth-1)

    unresolved=visit(0.,duration_s,policy.max_subdivisions)
    return EulerODEClearanceVerdict(
        "clear" if unresolved is None else "uncertain",
        visits,speed_bound,unresolved,
    )
