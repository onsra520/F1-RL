"""First finite-tyre sidewall contact cannot forge a vertical bump stop."""
from dataclasses import replace
import pytest
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.vehicle import VehicleState
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.rigid_body_impact_response import RigidBodyImpactMass
from src.simulation.wheel_edge_overtravel_impact import (
    incoming_edge,entry_margin,apply_edge_overtravel_impulse_at_prefix,
)
from src.rl.crash_terminal_registered import SOURCES

P=((-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.))
Q=Quaternion.identity()
Z=Vec3(0.,0.,0.)
ROAD=PlanarSurfacePatch("road",Z,Vec3(0.,0.,1.),Vec3(0.,1.,0.),2.,2.)
M=RigidBodyImpactMass(1000.,Vec3(500.,500.,500.))

def pose(x,y=-.3,vx=2.):
    return VehicleState(Vec3(x,y,0.),Vec3(vx,0.,0.),Q,Z,True)

def test_actual_sidewall_first_contact_impulse_loss_and_source():
    b=pose(-1.1001)
    a=pose(-1.0999)
    idx=incoming_edge(b,a,Z,ROAD,P,max_bump_m=.12)
    assert idx==3
    assert entry_margin(b,Z,P,idx)<0.
    assert entry_margin(a,Z,P,idx)>0.
    pre=pose(-1.10000000001)
    imp=apply_edge_overtravel_impulse_at_prefix(
        pre,M,wheel_local=Z,road=ROAD,polygon=P,edge_index=idx,
        max_bump_m=.12,first_clear_s=.00004999995,first_witness_s=.00005)
    assert imp.kind=="wheel_edge_hard_stop"
    assert imp.postimpact.position==imp.preimpact.position
    assert imp.postimpact.velocity.x==pytest.approx(0.)
    assert imp.impulse_n_s.x==pytest.approx(-2000.)
    assert imp.energy_lost_j==pytest.approx(2000.)
    assert SOURCES["wp3_grounded_wheel_edge_stop_crash_v4_dev"] == (
        "wheel_edge_hard_stop","grounded_wheel_edge_hard_stop_absorbing_v4_development")

@pytest.mark.parametrize(("first","last"),[
    (pose(-1.1),pose(-1.2)),     # retreat
    (pose(-1.3),pose(-1.2)),     # no entry
    (pose(-1.1001),pose(-1.0999,y=-.1)), # no overtravel
    (pose(-1.1001,vx=-2.),pose(-1.0999,vx=-2.)), # separating motion
    (pose(-1.1001,y=.2),pose(-1.0999,y=.2)),
])
def test_forged_footprint_sidewall_entry_rejected(first,last):
    with pytest.raises(ValueError):
        incoming_edge(first,last,Z,ROAD,P,max_bump_m=.12)

@pytest.mark.parametrize("x",[-1.2,-1.099,-1.0])
def test_unwitnessed_impact_prefix_is_not_accepted(x):
    with pytest.raises(ValueError):
        apply_edge_overtravel_impulse_at_prefix(
            pose(x),M,wheel_local=Z,road=ROAD,polygon=P,edge_index=3,
            max_bump_m=.12,first_clear_s=.00004999,first_witness_s=.00005)
