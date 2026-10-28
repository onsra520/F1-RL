"""Prospective frictionless rigid-body impact *response primitive*.

Given a separately witnessed, owner-certified contact plane and a contact
point on the car hull, derive normal impulse, angular impulse and kinetic
energy change. NO obstacle time-of-impact solver, no owner certification,
no actual scoring/terminal, and no simulator integration in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .body_collision import VehicleBodyGeometry
from .geometry.rotation import Quaternion
from .geometry.vector import Vec3
from .vehicle import VehicleState


def _finite_positive(v:object,key:str)->float:
    if type(v) not in (int,float) or not isfinite(v) or v<=0.:
        raise ValueError(f"{key} must be finite positive")
    return float(v)


@dataclass(frozen=True,slots=True)
class RigidBodyImpactMass:
    mass_kg:float
    inertia_body_kg_m2:Vec3

    def __post_init__(self):
        _finite_positive(self.mass_kg,"mass")
        if not isinstance(self.inertia_body_kg_m2,Vec3):
            raise ValueError("body-frame moment of inertia required")
        inertia=self.inertia_body_kg_m2
        for axis in (inertia.x,inertia.y,inertia.z):
            _finite_positive(axis,"principal inertia")
        if 2.*max(inertia.x,inertia.y,inertia.z)>inertia.x+inertia.y+inertia.z+1e-9:
            raise ValueError("inconsistent rigid-body inertia triangle")

    def world_inertia(self,q:Quaternion,w:Vec3)->Vec3:
        inv=Quaternion(q.w,-q.x,-q.y,-q.z)
        local=inv.rotate(w)
        i=self.inertia_body_kg_m2
        return q.rotate(Vec3(i.x*local.x,i.y*local.y,i.z*local.z))

    def world_inertia_inverse(self,q:Quaternion,h:Vec3)->Vec3:
        inv=Quaternion(q.w,-q.x,-q.y,-q.z)
        local=inv.rotate(h)
        i=self.inertia_body_kg_m2
        return q.rotate(Vec3(local.x/i.x,local.y/i.y,local.z/i.z))


@dataclass(frozen=True,slots=True)
class ImpactResponse:
    impulse_n_s:Vec3
    angular_impulse_nms:Vec3
    before_kinetic_j:float
    after_kinetic_j:float
    energy_lost_j:float
    pre_relative_normal_mps:float
    post_relative_normal_mps:float
    velocity_after_mps:Vec3
    angular_velocity_after_radps:Vec3
    response_version:str="frictionless_stationary_obb_impulse_v1"


def rigid_body_normal_impact(
    previous:VehicleState,
    geometry:VehicleBodyGeometry,
    mass:RigidBodyImpactMass,
    *,
    contact_point_world:Vec3,
    normal_obstacle_to_vehicle:Vec3,
    restitution:float=0.,
    face_tolerance_m:float=1e-6,
) -> ImpactResponse:
    if (not isinstance(previous,VehicleState) or
        not isinstance(geometry,VehicleBodyGeometry) or
        not isinstance(mass,RigidBodyImpactMass) or
        not isinstance(contact_point_world,Vec3) or
        not isinstance(normal_obstacle_to_vehicle,Vec3)):
        raise ValueError("typed vehicle/hull/mass/contact inputs required")
    if (type(restitution) not in (int,float) or not isfinite(restitution)
        or not 0.<=restitution<=1. or
        type(face_tolerance_m) not in (int,float) or not isfinite(face_tolerance_m)
        or not 0.<face_tolerance_m<=1e-3):
        raise ValueError("restitution and surface tolerance outside domain")
    q=previous.orientation
    normal=normal_obstacle_to_vehicle
    if abs(normal.magnitude-1.)>1e-8:
        raise ValueError("witness contact normal must be unit length")
    obb=geometry.box_at(previous.position,q)
    inverse=Quaternion(q.w,-q.x,-q.y,-q.z)
    relative=inverse.rotate(contact_point_world-obb.center)
    extents=obb.half_extents
    if (any(abs(c)>e+face_tolerance_m for c,e in (
        (relative.x,extents.x),(relative.y,extents.y),(relative.z,extents.z)))
        or min(abs(abs(c)-e) for c,e in (
            (relative.x,extents.x),(relative.y,extents.y),
            (relative.z,extents.z)))>face_tolerance_m):
        raise ValueError("contact point is not witnessed on the rigid body surface")
    r=contact_point_world-previous.position
    point_velocity=previous.velocity+previous.angular_velocity.cross(r)
    approaching=point_velocity.dot(normal)
    if approaching>=-1e-10:
        raise ValueError("stationary obstacle contact is separating or tangential")
    arm_cross_normal=r.cross(normal)
    denominator=(1./mass.mass_kg+
                 normal.dot(mass.world_inertia_inverse(q,arm_cross_normal).cross(r)))
    if not isfinite(denominator) or denominator<=0.:
        raise ValueError("impact effective mass is not physically positive")
    magnitude=-(1.+restitution)*approaching/denominator
    impulse=normal*magnitude
    torque_impulse=r.cross(impulse)
    v_after=previous.velocity+impulse/mass.mass_kg
    w_after=previous.angular_velocity+mass.world_inertia_inverse(q,torque_impulse)
    def kinetic(velocity:Vec3,angular:Vec3)->float:
        return (.5*mass.mass_kg*velocity.dot(velocity)+
                .5*angular.dot(mass.world_inertia(q,angular)))
    old,new=kinetic(previous.velocity,previous.angular_velocity),kinetic(v_after,w_after)
    scale=max(1.,old,new,impulse.magnitude*max(previous.speed,1.))
    if (not all(isfinite(x) for x in (magnitude,old,new)) or
        new>old+1e-9*scale):
        raise ValueError("impact would add unregistered kinetic energy")
    after_relative=(v_after+w_after.cross(r)).dot(normal)
    if abs(after_relative+restitution*approaching)>1e-8*max(1.,abs(approaching)):
        raise ValueError("normal restitution kinematics inconsistent")
    return ImpactResponse(
        impulse,torque_impulse,old,new,max(0.,old-new),approaching,
        after_relative,v_after,w_after)
