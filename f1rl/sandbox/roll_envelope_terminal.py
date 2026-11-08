"""Development-only, source-certified dynamics-envelope roll termination.

This is NOT a physical chassis/road collision. No normal impulse or collision
energy loss is invented. The event stops exactly at the last accepted RK4
state before the original authored roll domain boundary (0.7 rad).
"""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2,isfinite
from .vehicle import VehicleState
from .center_trajectory import CenterSegment
from .geometry.vector import Vec3

def _roll(state):
    right=state.orientation.rotate(Vec3(1.,0.,0.))
    up=state.orientation.rotate(Vec3(0.,1.,0.))
    return atan2(right.y,up.y)

VERSION="wp3_source_certified_tilt_envelope_terminal_v1_dev"

@dataclass(frozen=True,slots=True)
class RollEnvelopeSource:
    source_start:VehicleState
    source_rejected:VehicleState
    segment_start_s:float
    first_clear_local_s:float
    first_outside_local_s:float
    previous_poses:tuple[VehicleState,...]
    previous_times_s:tuple[float,...]
    previous_leaves:tuple[object,...]
    impact_prefix_audit:object
    body:object
    road_atlas:tuple

@dataclass(frozen=True,slots=True)
class RollEnvelopeStop:
    first_clear_s:float
    first_outside_s:float
    roll_limit_rad:float
    side:int
    initial_roll_rad:float
    clear_roll_rad:float
    rejected_roll_rad:float
    accepted:VehicleState
    rejected:VehicleState
    prefix:tuple[CenterSegment,...]
    source:object
    version:str=VERSION

    def __post_init__(self):
        nums=(self.first_clear_s,self.first_outside_s,self.roll_limit_rad,
              self.initial_roll_rad,self.clear_roll_rad,self.rejected_roll_rad)
        if (self.version!=VERSION
            or any(type(v) not in (int,float) or not isfinite(v) for v in nums)
            or self.side not in (-1,1) or type(self.side) is not int
            or not 0.<self.first_clear_s<self.first_outside_s
            or self.first_outside_s-self.first_clear_s>1.0001e-10
            or not 0.<self.roll_limit_rad<.75
            or not abs(self.initial_roll_rad)<self.roll_limit_rad
            or not 0.<self.side*self.clear_roll_rad<=self.roll_limit_rad
            or not self.side*self.rejected_roll_rad>self.roll_limit_rad
            or not isinstance(self.accepted,VehicleState)
            or not isinstance(self.rejected,VehicleState)
            or not isinstance(self.prefix,tuple) or not self.prefix
            or self.prefix[-1].end!=self.accepted.position):
            raise ValueError("unwitnessed or forged roll-envelope exit")
        if (abs(_roll(self.accepted)-self.clear_roll_rad)>1e-9
            or abs(_roll(self.rejected)-self.rejected_roll_rad)>1e-9):
            raise ValueError("roll-envelope source attitudes do not agree")


class CertifiedRollEnvelopeExit(RuntimeError):
    def __init__(self,stop:RollEnvelopeStop):
        super().__init__("certified roll model-envelope termination (not a physical crash)")
        self.stop=stop
