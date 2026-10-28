"""Audited, deliberately dissipative zero-unsprung-mass spring release.

A disappearing road-supported contact may discard its spring potential into
a modeled road/deformation heat sink ONLY at a bounded unilateral departure.
No change of kinetic body momentum is permitted at this *release* event;
all gains, new contacts, missing brackets and numerical energy violations
fail closed. This is a development contact policy, not owner calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True,slots=True)
class WheelEdgeReleaseEnergy:
    removed_supports:tuple[int,...]
    transferred_to_environment_j:float
    residual_after_transfer_j:float
    bracket_duration_s:float
    source_version:str="wp3_road_edge_dissipative_release_v1"


def certify_wheel_edge_release_energy(
    masks:tuple[frozenset[int],...],
    *,
    start_spring_energy_j:float,
    end_spring_energy_j:float,
    balance_residual_j:float,
    balance_tolerance_j:float,
    bracket_duration_s:float,
    maximum_event_duration_s:float=1e-8,
)->WheelEdgeReleaseEnergy:
    if (not isinstance(masks,tuple) or len(masks)<2
        or any(type(x) is not frozenset for x in masks)
        or any(type(index) is not int or index not in range(4)
               for group in masks for index in group)):
        raise ValueError("required four-wheel source-typed masks")
    fields=(start_spring_energy_j,end_spring_energy_j,balance_residual_j,
            balance_tolerance_j,bracket_duration_s,maximum_event_duration_s)
    if (any(type(x) not in (float,int) or not isfinite(x) for x in fields)
        or start_spring_energy_j<0. or end_spring_energy_j<0.
        or balance_tolerance_j<=0. or bracket_duration_s<=0.
        or maximum_event_duration_s<=0. or
        bracket_duration_s>maximum_event_duration_s):
        raise ValueError("release requires finite positive event window and energy")
    if any(not right.issubset(left) for left,right in zip(masks,masks[1:])):
        raise ValueError("recontact/increasing support is not dissipative edge release")
    removed=masks[0]-masks[-1]
    if not removed:
        raise ValueError("no genuinely released wheel exists in event bracket")
    liberated=start_spring_energy_j-end_spring_energy_j
    if liberated<=0.:
        raise ValueError("no physically stored spring potential lost")
    after=balance_residual_j+liberated
    if abs(after)>balance_tolerance_j:
        raise ValueError("missing energy balance; cannot invent arbitrary dissipation")
    return WheelEdgeReleaseEnergy(tuple(sorted(removed)),liberated,after,
                                  bracket_duration_s)
