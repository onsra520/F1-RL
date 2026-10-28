"""Prospective continuous centre-reference checkpoint/finish crossing (5A).

The gate's orientation local +Z defines *allowed forward direction*. A
crossing is reported only when the centre transitions from negative local Z
to >=0, intersecting the gate's finite local XY aperture. Gate orientation
must be authored and reviewed, not learned or inferred from legacy reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume


@dataclass(frozen=True,slots=True)
class DirectedCrossing:
    fraction: float
    center: Vec3

    def __post_init__(self) -> None:
        if not isinstance(self.center, Vec3):
            raise ValueError("crossing center must be Vec3")
        if type(self.fraction) not in (int, float) or not isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError("crossing fraction must be in [0,1]")


def directed_center_gate_crossing(
    previous: Vec3,
    candidate: Vec3,
    gate: OrientedBoxVolume,
    *,
    tolerance_m: float = 1e-9,
) -> DirectedCrossing | None:
    """Return forward crossing for a finite, rotated checkpoint gate.

    The 3D segment is linear between accepted physics positions; no
    tunnelling across arbitrarily thin gates. This function does not
    decide checkpoint ordering or finish authorization; those are
    separate race-state-machine obligations.
    """
    if not isinstance(previous, Vec3) or not isinstance(candidate, Vec3) or not isinstance(gate, OrientedBoxVolume):
        raise ValueError("crossing requires two Vec3 and OrientedBoxVolume")
    if (
        type(tolerance_m) not in (int,float)
        or not isfinite(tolerance_m) or tolerance_m < 0 or tolerance_m > 1e-4
    ):
        raise ValueError("crossing tolerance must be a small finite nonnegative number")

    q=gate.orientation
    forward=q.rotate(Vec3(0.,0.,1.))
    right=q.rotate(Vec3(1.,0.,0.))
    up=q.rotate(Vec3(0.,1.,0.))
    s0=(previous-gate.center).dot(forward)
    s1=(candidate-gate.center).dot(forward)
    # Positive forward travel; 'already on positive side' cannot retrigger.
    # Strict negative starting side prevents repeated event at plane origin.
    if s0>=-tolerance_m or s1 < -tolerance_m or s1<=s0:
        return None
    fraction=-s0/(s1-s0)
    if not 0.0<=fraction<=1.0:
        return None
    center=previous+(candidate-previous)*fraction
    delta=center-gate.center
    if abs(delta.dot(right))>gate.half_extents.x+tolerance_m:
        return None
    if abs(delta.dot(up))>gate.half_extents.y+tolerance_m:
        return None
    return DirectedCrossing(fraction,center)
