"""Opt-in per-wheel spatial road-edge contact *classification* (not scorer).

A wheel outside every authored 2.5D patch is geometry-released; a wheel
above the droop stop is unloaded, but not falsely counted as geometry exit.
Penetration beyond bump limit, atlas seam contradictions, and ambiguity fail
closed. The inertial 6-DOF solver consumes only witnessed contacts. This
sampler alone does not certify landing or official outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .geometry.vector import Vec3
from .independent_suspension_dynamics import SuspensionResolutionError
from .multi_surface_support import (
    IndependentWheelContact,_atlas,_within_shape,_height,
)
from .vehicle import VehicleState

_NAMES=("front_left","front_right","rear_left","rear_right")


@dataclass(frozen=True,slots=True)
class SpatialWheelSupportSample:
    contacts:tuple[IndependentWheelContact|None,...]
    classification:tuple[str,...]
    version:str="spatial_partial_wheels_v4"

    def __post_init__(self):
        if len(self.contacts)!=4 or len(self.classification)!=4:
            raise ValueError("four individually classified wheel references required")
        if any(c not in ("supported","outside_road","unloaded_above_droop")
               for c in self.classification):
            raise ValueError("invalid wheel support category")
        if any((c is None)!=(label!="supported")
               for c,label in zip(self.contacts,self.classification,strict=True)):
            raise ValueError("support category/witness disagreement")

    @property
    def support_count(self)->int:
        return sum(c is not None for c in self.contacts)


def sample_spatial_partial_wheels(
    vehicle:VehicleState,contact_policy,atlas
)->SpatialWheelSupportSample:
    if not isinstance(vehicle,VehicleState) or atlas is None or len(atlas)!=2:
        raise ValueError("vehicle and consistent authored road atlas required")
    patches,polygons=atlas
    contacts=[]
    classifications=[]
    for wheel_index,(name,local) in enumerate(zip(_NAMES,contact_policy.wheels.positions,strict=True)):
        point=vehicle.position+vehicle.orientation.rotate(local)
        candidates=[]
        for patch,polygon in zip(patches,polygons,strict=True):
            if not _within_shape(point.x,point.z,polygon):continue
            height=_height(patch,point.x,point.z)
            deflection=height-point.y
            if not isfinite(deflection):
                raise SuspensionResolutionError("nonfinite per-wheel road gap")
            if deflection>contact_policy.max_bump_m+1e-9:
                error=SuspensionResolutionError(
                    f'spatial partial wheel over bump: wheel={name} patch={patch.id} '
                    f'deflection_m={deflection:.9g} max_bump_m={contact_policy.max_bump_m:.9g} '
                    f'vehicle_position={vehicle.position} velocity={vehicle.velocity}',
                    failure_code="over_bump_travel")
                # A witness to the *failed candidate*, never an accepted state.
                error.wheel_index=wheel_index
                error.wheel_local=local
                error.road_patch=patch
                error.rejected_vehicle=vehicle
                raise error
            candidates.append((abs(deflection),patch.id,height,deflection))
        if not candidates:
            contacts.append(None)
            classifications.append("outside_road")
            continue
        _,patch_id,height,deflection=min(candidates)
        if deflection < -contact_policy.max_droop_m-1e-9:
            contacts.append(None)
            classifications.append("unloaded_above_droop")
        else:
            contacts.append(IndependentWheelContact(name,patch_id,height,deflection))
            classifications.append("supported")
    return SpatialWheelSupportSample(tuple(contacts),tuple(classifications))
