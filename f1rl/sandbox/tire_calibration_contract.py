"""WP7 tire geometry and physical measurement provenance: explicit opt-in.

A tire contact point is NOT a wheel suspension reference. This immutable
contract defines wheel HUBS and a spherical rolling-radius approximation,
with metadata requiring separately traceable measurements. Legacy contact
point geometry is not silently reinterpreted as hubs, and this module does
not claim Pacejka/contact-patch or measured physical calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .geometry.vector import Vec3
from .geometry.rotation import Quaternion

@dataclass(frozen=True,slots=True)
class TireHubGeometry:
    hub_local_positions: tuple[Vec3,Vec3,Vec3,Vec3]
    rolling_radius_m: float
    nominal_width_m: float

    def __post_init__(self):
        if (not isinstance(self.hub_local_positions,tuple)
                or len(self.hub_local_positions)!=4
                or any(not isinstance(p,Vec3) for p in self.hub_local_positions)
                or len(set(self.hub_local_positions))!=4):
            raise ValueError("four distinct wheel hub centres required")
        for key in ("rolling_radius_m","nominal_width_m"):
            v=getattr(self,key)
            if type(v) not in (int,float) or not isfinite(v) or v<=0.:
                raise ValueError(key+" must be finite positive")
        if self.rolling_radius_m>1. or self.nominal_width_m>1.:
            raise ValueError("wheel geometry outside registered measurement domain")


@dataclass(frozen=True,slots=True)
class TireContactSample:
    wheel_index: int
    patch_id: str
    hub_world: Vec3
    contact_world: Vec3
    signed_surface_gap_m: float
    normal: Vec3


def sample_spherical_tire_contact(
    origin: Vec3,
    orientation: Quaternion,
    wheel_index: int,
    patch,
    geometry: TireHubGeometry,
)->TireContactSample:
    """Sphere-vs-plane candidate in 3D, not an accepted finite-patch contact.

    Caller must *separately* verify the projected contact is inside the
    finite road patch and wheel/body swept geometry is collision-free.
    """
    from .track.patch import PlanarSurfacePatch
    if (not isinstance(origin,Vec3) or not isinstance(orientation,Quaternion)
            or type(wheel_index) is not int or not 0<=wheel_index<4
            or not isinstance(patch,PlanarSurfacePatch)
            or not isinstance(geometry,TireHubGeometry)):
        raise ValueError("registered hub/plane tire inputs required")
    hub=origin+orientation.rotate(geometry.hub_local_positions[wheel_index])
    normal=patch.normal
    contact=hub-normal*geometry.rolling_radius_m
    gap=(hub-patch.origin).dot(normal)-geometry.rolling_radius_m
    return TireContactSample(
        wheel_index,patch.id,hub,contact,gap,normal,
    )


@dataclass(frozen=True,slots=True)
class MeasuredPhysicsParameter:
    name: str
    value: float
    unit: str
    absolute_uncertainty: float
    evidence_id: str
    method: str

    def __post_init__(self):
        if any(not isinstance(s,str) or not s.strip()
               for s in (self.name,self.unit,self.evidence_id,self.method)):
            raise ValueError("measurement requires traceable identifiers, unit and method")
        if (type(self.value) not in (int,float) or not isfinite(self.value)
                or type(self.absolute_uncertainty) not in (int,float)
                or not isfinite(self.absolute_uncertainty)
                or self.absolute_uncertainty<0.):
            raise ValueError("measurement and uncertainty must be finite")
        if self.value<=0.:
            raise ValueError("measured physical parameter must be positive")


@dataclass(frozen=True,slots=True)
class PhysicsCalibrationRegister:
    measurements: tuple[MeasuredPhysicsParameter,...]
    verifier_evidence_id: str

    def __post_init__(self):
        if (not self.verifier_evidence_id or
                len({m.name for m in self.measurements})!=len(self.measurements)):
            raise ValueError("independent verifier and unique measurements required")
        if any(not isinstance(m,MeasuredPhysicsParameter) for m in self.measurements):
            raise ValueError("only evidenced measurements may enter register")

    def missing(self,required: tuple[str,...])->tuple[str,...]:
        recorded={m.name for m in self.measurements}
        return tuple(name for name in required if name not in recorded)

REQUIRED_WP7_MEASUREMENTS=(
    "mass_kg","pitch_inertia_kg_m2","yaw_inertia_kg_m2",
    "roll_inertia_kg_m2","wheel_radius_m","wheel_width_m",
    "spring_stiffness_npm","damper_nspm",
    "front_cornering_stiffness_nprad","rear_cornering_stiffness_nprad",
    "tire_friction_mu","tire_restitution",
)
