"""Timed accepted center trajectories: numerical linear substeps or exact gravity.

These laws describe the registered integrator path. They do not certify body
clearance, physical contact, or scientific approval; physics owns those gates.
"""
from dataclasses import dataclass
from math import copysign, isfinite, sqrt, ulp

from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume


@dataclass(frozen=True, slots=True)
class CenterSegment:
    start_time_s: float
    end_time_s: float
    start: Vec3
    end: Vec3
    initial_velocity: Vec3 | None = None
    gravity_mps2: float = 0.

    def __post_init__(self):
        if (any(type(t) not in (int,float) or not isfinite(t)
                for t in (self.start_time_s,self.end_time_s,self.gravity_mps2))
                or self.start_time_s < 0. or self.end_time_s <= self.start_time_s
                or self.gravity_mps2 < 0.
                or not isinstance(self.start,Vec3) or not isinstance(self.end,Vec3)):
            raise ValueError("finite ordered center trajectory required")
        if (self.initial_velocity is None) != (self.gravity_mps2 == 0.):
            raise ValueError("ballistic segment requires velocity and positive gravity")
        if self.initial_velocity is not None:
            if not isinstance(self.initial_velocity,Vec3):
                raise ValueError("ballistic velocity must be Vec3")
            if (self.position_at(self.end_time_s)-self.end).magnitude > 1e-7:
                raise ValueError("ballistic endpoint disagrees with exact gravity law")

    def position_at(self, time_s):
        if (type(time_s) not in (int,float) or not isfinite(time_s)
                or not self.start_time_s <= time_s <= self.end_time_s):
            raise ValueError("time outside center segment")
        t = time_s-self.start_time_s
        if self.initial_velocity is not None:
            return self.start+self.initial_velocity*t+Vec3(0.,-.5*self.gravity_mps2*t*t,0.)
        return self.start+(self.end-self.start)*(t/(self.end_time_s-self.start_time_s))

    def directed_crossings(self, gate: OrientedBoxVolume):
        """Exact positive-direction plane roots, checked against the 3D aperture."""
        if not isinstance(gate,OrientedBoxVolume):
            raise ValueError("gate must be OrientedBoxVolume")
        duration = self.end_time_s-self.start_time_s
        forward = gate.orientation.rotate(Vec3(0.,0.,1.))
        c = (self.start-gate.center).dot(forward)
        velocity = self.initial_velocity or (self.end-self.start)/duration
        b = velocity.dot(forward)
        a = -.5*self.gravity_mps2*forward.y
        if not all(isfinite(v) for v in (a,b,c,b*b-4*a*c)):
            raise ValueError("gate root coefficients exceed finite trajectory domain")
        if a == 0.:
            roots = (-c/b,) if b > 0. else ()
        else:
            discriminant = b*b-4*a*c
            if discriminant <= 0.:
                roots = ()
            else:
                q = -.5*(b+copysign(sqrt(discriminant),b))
                roots = (q/a,c/q) if q != 0. else ()
        result = []
        for t in sorted(set(roots)):
            # Division can put an endpoint root one ULP beyond the interval.
            # Clamp only roundoff-sized overshoot when the actual endpoint
            # has reached the forward side; never extend the physical path.
            if (duration < t <= duration+8*ulp(duration)
                    and (self.end-gate.center).dot(forward) >= 0.):
                t = duration
            # Exclude the starting plane to prevent replay on the next tick.
            if not 0. < t <= duration or b+2*a*t <= 1e-12:
                continue
            point = self.position_at(self.start_time_s+t)-gate.center
            if all(abs(point.dot(gate.orientation.rotate(axis))) <= half+1e-9
                   for axis,half in ((Vec3(1.,0.,0.),gate.half_extents.x),
                                     (Vec3(0.,1.,0.),gate.half_extents.y))):
                result.append(self.start_time_s+t)
        return tuple(result)


def linear_center_segments(poses, start_time_s, end_time_s):
    """Suspension poses are equally spaced in the numerical integration time."""
    if len(poses) < 2 or end_time_s <= start_time_s:
        raise ValueError("nonempty timed integration path required")
    h = (end_time_s-start_time_s)/(len(poses)-1)
    return tuple(CenterSegment(start_time_s+i*h,start_time_s+(i+1)*h,a.position,b.position)
                 for i,(a,b) in enumerate(zip(poses,poses[1:])))
