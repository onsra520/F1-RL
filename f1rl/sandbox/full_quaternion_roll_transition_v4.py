"""A source-bound *kinematic* large-roll transition for full-quaternion Physics v4.

Crossing the legacy small-angle envelope is NOT an impact. The physically
accepted v4 equations integrate the complete quaternion and world angular
momentum with RK4; any later chassis/road contact remains independently
certified via bounded finite-patch swept SAT. This certificate proves a
unique, monotone roll-envelope crossing on the already accepted short
quaternion chord. It does not create a fictitious contact, impulse, energy
loss, finish, or registered scientific claim.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2,isfinite
from .body_collision import _slerp
from .geometry.vector import Vec3
from .vehicle import VehicleState

RIGHT=Vec3(1.,0.,0.)
UP=Vec3(0.,1.,0.)
FORWARD=Vec3(0.,0.,1.)

def _roll(q):
    right=q.rotate(RIGHT)
    up=q.rotate(UP)
    return atan2(right.y,up.y)

@dataclass(frozen=True,slots=True)
class FullQuaternionRollTransition:
    first_clear_s:float
    first_crossing_s:float
    boundary_rad:float
    sign:int
    start_roll_rad:float
    end_roll_rad:float
    schema_version:str="v4_full_quaternion_roll_envelope_crossing_v1"

def certify_full_quaternion_roll_transition(
    before:VehicleState,after:VehicleState,*,bound_rad:float,
    duration_s:float,resolution_s:float=1e-10
)->FullQuaternionRollTransition|None:
    if (not isinstance(before,VehicleState) or not isinstance(after,VehicleState)
        or any(type(x) not in (int,float) or not isfinite(x)
               for x in (bound_rad,duration_s,resolution_s))
        or not 0.<bound_rad<.75 or not 0.<resolution_s<duration_s):
        raise ValueError("finite source-bound full-quaternion roll inputs required")
    r0,r1=_roll(before.orientation),_roll(after.orientation)
    if abs(r0)>=bound_rad or abs(r1)<bound_rad:
        return None
    sign=1 if r1>0 else -1
    if sign*r0<0. or sign*r1<bound_rad:
        raise ValueError("roll envelope crossing has ambiguous direction")
    # The shortest-arc world rotation between accepted source orientations.
    q0,q1=before.orientation,after.orientation
    delta=q1*type(q0)(q0.w,-q0.x,-q0.y,-q0.z)
    if delta.w<0.:
        delta=type(delta)(-delta.w,-delta.x,-delta.y,-delta.z)
    raw=Vec3(delta.x,delta.y,delta.z)
    if raw.magnitude<1e-12:
        raise ValueError("rotational chord has no finite crossing axis")
    axis=raw/raw.magnitude
    theta=2.*atan2(raw.magnitude,delta.w)
    if theta<=1e-10 or theta>.12:
        raise ValueError("roll transition requires a short full-quaternion RK4 leaf")
    qmid=_slerp(q0,q1,.5)
    forward=qmid.rotate(FORWARD)
    # EXACT derivative of roll(t)=atan2(Ry,Uy), with world-fixed SLERP axis:
    # droll/dt = theta*(axis.x*Fx + axis.z*Fz)/(1-Fy**2).
    # |Fy(t)-Fy(mid)| <= theta*|t-.5|. The following is a rigorous
    # strictly-positive numerator bound over the COMPLETE source chord,
    # rather than a few possibly misleading interior samples.
    numerator=axis.x*forward.x+axis.z*forward.z
    if sign*numerator<=abs(axis.y)*theta/2.+1e-8:
        raise ValueError("uniqueness of large-roll crossing not proven")
    if abs(forward.y)+theta/2.>=.95:
        raise ValueError("quaternion roll coordinates near singular pitch")
    def signed_margin(t):
        return sign*_roll(_slerp(q0,q1,t))-bound_rad
    if signed_margin(0.)>=0. or signed_margin(1.)<0.:
        raise ValueError("crossing does not bracket the tilt envelope")
    lo,hi=0.,1.
    for _ in range(64):
        if (hi-lo)*duration_s<=resolution_s:
            break
        mid=(lo+hi)/2.
        if signed_margin(mid)<0.:
            lo=mid
        else:
            hi=mid
    if (not 0.<=lo<hi<=1.
        or hi*duration_s-lo*duration_s>resolution_s+1e-14
        or signed_margin(lo)>1e-9 or signed_margin(hi)<-1e-9):
        raise ValueError("large-roll crossing bracket unresolved")
    return FullQuaternionRollTransition(
        lo*duration_s,hi*duration_s,bound_rad,sign,r0,r1)

def verify_full_quaternion_roll_transition(
    before:VehicleState,after:VehicleState,saved:FullQuaternionRollTransition|None,
    *,bound_rad:float,duration_s:float
)->bool:
    """Recompute source geometry, rejecting forged timing, thresholds and rolls.

    1 ps is solely the machine-roundoff budget for original RK4 h versus
    differences between large absolute segment timestamps.
    """
    expected=certify_full_quaternion_roll_transition(
        before,after,bound_rad=bound_rad,duration_s=duration_s)
    if saved is None or expected is None:
        return saved is None and expected is None
    return (isinstance(saved,FullQuaternionRollTransition)
            and saved.schema_version==expected.schema_version
            and saved.boundary_rad==expected.boundary_rad
            and saved.sign==expected.sign
            and saved.start_roll_rad==expected.start_roll_rad
            and saved.end_roll_rad==expected.end_roll_rad
            and abs(saved.first_clear_s-expected.first_clear_s)<=1e-12
            and abs(saved.first_crossing_s-expected.first_crossing_s)<=1e-12
            and 0.<saved.first_crossing_s-saved.first_clear_s<=1.0001e-10)
