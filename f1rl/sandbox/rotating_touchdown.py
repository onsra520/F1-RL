"""Conservative rotating wheel/plane down-crossing localization.

Supports only research-restricted fixed-world angular-velocity trajectories,
height-consistent 2.5D planar road patches, and wheel-reference points. This is
not an ODE-certified multi-contact manifold. Inconclusive sign/coverage
intervals are rejected, never interpreted as a safe airborne episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .angular_landing_impulse import AngularLandingImpulsePolicy, ImpulseResolutionError
from .geometry.vector import Vec3
from .multi_surface_support import _within_shape
from .road_footprint import _hull
from .multi_surface_support import _intersection, _square


@dataclass(frozen=True, slots=True)
class RotatingWheelTouchdown:
    time_s: float
    wheel_index: int
    patch_id: str

    def __post_init__(self) -> None:
        if (not isfinite(self.time_s) or self.time_s < 0.
                or type(self.wheel_index) is not int
                or not 0 <= self.wheel_index < 4 or not self.patch_id):
            raise ValueError("invalid rotating wheel event")


def first_rotating_wheel_touchdown(
    pose_at, wheels, atlas, dt: float, gravity: float,
    policy: AngularLandingImpulsePolicy,
    *, certify_first_root: bool = False,
) -> RotatingWheelTouchdown | None:
    """Find first conservative down-crossing, or fail if any gap is unresolved.

    The recursive interval verifier uses a Lipschitz bound of the signed
    gap function. If endpoints show no crossing and the bound cannot exclude
    an interior crossing, recursively subdivide. At a detected sign-change,
    use bounded bisection. Interior-to-exterior polygon transitions near
    the plane are explicitly unsupported.
    """
    if not isinstance(policy, AngularLandingImpulsePolicy):
        raise ValueError("explicit angular event detection policy required")
    if type(certify_first_root) is not bool:
        raise ValueError("certify_first_root must be bool")
    if (type(dt) not in (int,float) or not isfinite(dt) or dt <= 0.
            or type(gravity) not in (int,float) or not isfinite(gravity) or gravity <= 0.):
        raise ValueError("positive finite flight timestep and gravity required")
    patches, polygons = atlas
    if len(patches) != len(polygons):
        raise ValueError("invalid atlas")
    eps=1e-8
    segments=policy.event_bracket_segments
    h=dt/segments
    event_candidates=[]
    for patch, polygon in zip(patches, polygons, strict=True):
        normal=patch.normal
        if normal.y < .5:
            raise ImpulseResolutionError("rotating plane events require upward bounded slope")
        for i,w in enumerate(wheels.positions):
            wheel_radius=w.magnitude
            def data(t):
                state=pose_at(t)
                point=state.position + state.orientation.rotate(w)
                gap=(point-patch.origin).dot(normal)
                inside=_within_shape(point.x,point.z,polygon)
                return point,gap,inside,state
            def interval(lo,hi,depth):
                p0,g0,in0,st0=data(lo)
                p1,g1,in1,st1=data(hi)
                if in0 and g0 < -eps:
                    raise ImpulseResolutionError("wheel starts below road during rotating flight")
                if in1 and g1 < -eps and not in0:
                    raise ImpulseResolutionError("wheel enters patch already under road surface")
                if in0 != in1:
                    # A wheel can enter/exit the finite road polygon within an
                    # unresolved substep. Without a certified edge-impact
                    # manifold this must not be treated as clean flight.
                    if min(g0,g1) < (abs(st0.velocity.dot(normal))+gravity*normal.y*dt
                                     +st0.angular_velocity.magnitude*wheel_radius)*(hi-lo)+eps:
                        raise ImpulseResolutionError("rotating wheel polygon crossing near road is unsupported")
                    return None
                if not in0:
                    # Conservative swept XY ground-projection disc. If the
                    # entire swept wheel footprint misses this road, no
                    # contact with it is possible. Otherwise recurse/abort.
                    omega=st0.angular_velocity.magnitude
                    rad=omega*wheel_radius*(hi-lo)
                    ext=_hull((*_square(p0.x,p0.z,rad+1e-9),
                               *_square(p1.x,p1.z,rad+1e-9)))
                    if not _intersection(ext,polygon):
                        return None
                    if depth == 0:
                        raise ImpulseResolutionError("rotating wheel road-edge coverage unresolved")
                    mid=(lo+hi)/2.
                    return interval(lo,mid,depth-1) or interval(mid,hi,depth-1)
                start_speed=(st0.velocity+st0.angular_velocity.cross(
                    p0-st0.position)).dot(normal)
                if g0 <= eps and start_speed <= 0.:
                    raise ImpulseResolutionError("initial rotating wheel is in descending road contact")
                if g0 <= eps and start_speed > 0.:
                    # Immediate outward departure after an impulse is not
                    # a second collision. Certify monotonic separation over
                    # this interval from a bound on d(v_contact_n)/dt.
                    max_rate_change=(gravity*normal.y
                                     +st0.angular_velocity.magnitude**2*wheel_radius)
                    if start_speed-max_rate_change*(hi-lo) > eps:
                        return None
                if g0 > eps and g1 <= 0.:
                    # A sign change alone does not establish the first root
                    # of a rotating gap. Certify decreasing motion or refine
                    # left-to-right, rejecting exhausted uncertainty.
                    rate_bound = gravity*normal.y + st0.angular_velocity.magnitude**2*wheel_radius
                    if certify_first_root and start_speed+rate_bound*(hi-lo) >= 0.:
                        if depth == 0:
                            raise ImpulseResolutionError("first-root monotonicity unresolved")
                        mid = (lo+hi)/2.
                        return interval(lo,mid,depth-1) or interval(mid,hi,depth-1)
                    a,b=lo,hi
                    for _ in range(policy.event_bisections):
                        if b-a <= policy.event_time_tolerance_s:
                            break
                        mid=(a+b)/2.
                        _,gm,inside,_=data(mid)
                        if not inside:
                            raise ImpulseResolutionError("wheel leaves patch during root refinement")
                        if gm > 0.:a=mid
                        else:b=mid
                    if certify_first_root and b-a > policy.event_time_tolerance_s:
                        raise ImpulseResolutionError("event-time tolerance not met within bisection budget")
                    # Emit the non-penetrating side of the time bracket.
                    # Returning b would start a rebound microscopically
                    # below the road, violating the next frame's hard gate.
                    return RotatingWheelTouchdown(a,i,patch.id)
                if g1 < -eps:
                    raise ImpulseResolutionError("rotating wheel penetrated without localized touchdown")
                # In nearest-endpoint distance <= h/2, signed gap rate has
                # bound |v_n| + g*dt + |omega|*wheel_arm. Require positive
                # margin, otherwise bisect until decisive or fail closed.
                L=(abs(st0.velocity.dot(normal))+gravity*normal.y*dt
                   +st0.angular_velocity.magnitude*wheel_radius)
                if min(g0,g1) > L*(hi-lo)/2. + eps:
                    return None
                if depth == 0:
                    raise ImpulseResolutionError("rotating contact timing remains unresolved")
                mid=(lo+hi)/2.
                return interval(lo,mid,depth-1) or interval(mid,hi,depth-1)
            for segment in range(segments):
                t0=segment*h
                earliest=min((c.time_s for c in event_candidates),default=dt)
                # Earlier wheel event uses the *above-plane* bisection
                # endpoint. Inspect the paired time-tolerance bracket for
                # co-contacting wheels before concluding their root absent.
                # This does not advance the simulated chassis past event time.
                discovery_limit=min(dt,earliest+2.*policy.event_time_tolerance_s)
                if t0 >= discovery_limit:
                    break
                t1=min((segment+1)*h,discovery_limit)
                found=interval(t0,t1,policy.event_bisections)
                if found is not None:
                    event_candidates.append(found)
                    break
    return min(event_candidates,key=lambda e:(e.time_s,e.patch_id,e.wheel_index)) if event_candidates else None
