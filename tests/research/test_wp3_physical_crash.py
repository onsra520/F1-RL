"""First physically applied obstacle crash, before later checkpoints."""
from types import SimpleNamespace
import pytest
from src.sandbox.physical_crash import first_applied_grounded_crash
from src.sandbox.center_trajectory import CenterSegment
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.rigid_body_impact_response import RigidBodyImpactMass
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState

Q=Quaternion.identity()
ZERO=Vec3(0.,0.,0.)
BODY=VehicleBodyGeometry(.4,.4,.4,ZERO)
MASS=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))
OBS=SimpleNamespace(id="obstacle-x",volume=OrientedBoxVolume(ZERO,Vec3(.1,2.,2.),Q))

def pose(x, vx=-2.):
    return VehicleState(Vec3(x,0.,0.),Vec3(vx,0.,0.),Q,ZERO,True)


def test_applied_first_crash_has_real_changed_vehicle_momentum():
    a,b=pose(.6),pose(0.)
    seg=(CenterSegment(0.,.3,a.position,b.position),)
    hit=first_applied_grounded_crash((a,b),seg,(OBS,),BODY,MASS,fixed_dt_s=.3)
    assert hit is not None
    assert hit.obstacle_id=="obstacle-x"
    assert hit.version=="wp3_grounded_applied_crash_v4_dev"
    assert .1499<hit.last_proven_clear_s<hit.tick_contact_time_s<.1501
    assert hit.contact.postimpact.velocity.x>hit.contact.preimpact.velocity.x
    assert hit.contact.energy_lost_j>0.
    assert hit.prefix[-1].start==hit.prefix[-1].end
    assert hit.prefix[-1].end_time_s==.3


def test_no_contact_has_no_crash():
    a,b=pose(.8),pose(.7)
    assert first_applied_grounded_crash((a,b),(
        CenterSegment(0.,.05,a.position,b.position),),
        (OBS,),BODY,MASS,fixed_dt_s=.05) is None


def test_forged_ballistic_crash_cannot_use_grounded_certificate():
    a,b=pose(.6),pose(0.)
    with pytest.raises(ValueError,match="distinct ballistic"):
        first_applied_grounded_crash((a,b),(
          CenterSegment(0.,.3,a.position,b.position,
                        initial_velocity=Vec3(-2.,1.47,0.),gravity_mps2=9.8),),
          (OBS,),BODY,MASS,fixed_dt_s=.3)
