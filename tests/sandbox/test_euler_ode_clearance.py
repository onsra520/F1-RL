"""Analytic Euler torque-free clearance enclosure vs numerical RK4 CCD.

Every 'clear' result follows a conservative energy-derived angular-speed
bound, NOT trust in RK4 accuracy. Unresolved intervals are not contacts.
"""
from dataclasses import replace
import pytest

from src.sandbox.euler_ode_clearance import (
    EulerODEEnclosurePolicy,certify_torque_free_ode_clearance,
)
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.body_collision import VehicleBodyGeometry
from src.sandbox.multi_surface_support import _atlas
from src.sandbox.ramp_support import WheelContactPoints
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.volume import OrientedBoxVolume
from src.sandbox.vehicle import VehicleState
from src.sandbox.torque_free_rotation import TorqueFreeRotationPolicy
from src.sandbox.torque_free_swept_ccd import TorqueFreeSweepPolicy

ROAD=PlanarSurfacePatch("plane",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,20.)
ATLAS=_atlas((ROAD,),1e-6)
BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
ANGULAR=TorqueFreeRotationPolicy(1800.,1600.,600.,integration_steps=32)
POLICY=EulerODEEnclosurePolicy(max_subdivisions=12)


def airborne(*,pos=Vec3(0.,5.,0.),vel=Vec3(1.,0.,0.),
             spin=Vec3(.15,.24,.17)):
    return VehicleState(pos,vel,Quaternion.identity(),spin,False)

def prove(start=None, *, obstacles=(),duration=.2,policy=POLICY):
    return certify_torque_free_ode_clearance(
        start or airborne(),ANGULAR,BODY,W,ATLAS,obstacles,
        duration_s=duration,gravity_mps2=9.81,policy=policy,
    )

def box(pos=Vec3(5.,5.,0.)):
    return OrientedBoxVolume(pos,Vec3(.2,.3,.2),Quaternion.identity())

def test_far_road_and_obstacle_proved_clear_despite_unknown_rk4_error():
    v=prove(obstacles=(box(),))
    assert v.valid
    assert v.first_uncertain_window_s is None
    assert v.angular_speed_bound_radps>0.
    assert v.checks>0.

def test_fast_midstep_obstacle_tunnel_cannot_pass_analytic_enclosure():
    start=airborne(pos=Vec3(-5.,5.,0.),vel=Vec3(50.,0.,0.),
                   spin=Vec3(0.,0.,0.))
    v=prove(start,obstacles=(box(Vec3(0.,5.,0.)),))
    assert not v.valid
    assert v.status=="uncertain"
    lo,hi=v.first_uncertain_window_s
    assert lo<.1<hi or abs(lo-.1)<.02

def test_road_wheel_proximity_is_uncertain_never_falsely_safe():
    near=airborne(pos=Vec3(0.,.3,0.),vel=Vec3(0.,0.,0.),
                  spin=Vec3(0.,0.,0.))
    v=prove(near,duration=.05)
    assert not v.valid
    assert v.first_uncertain_window_s is not None

def test_low_clearance_but_static_body_above_road_can_be_proved():
    still=airborne(pos=Vec3(0.,1.1,0.),vel=Vec3(0.,0.,0.),
                   spin=Vec3(0.,0.,0.))
    result=prove(still,duration=.1)
    assert result.valid

def test_increasing_angular_speed_does_not_shrink_analytic_uncertainty():
    slow=airborne(pos=Vec3(0.,1.1,0.),vel=Vec3(0.,0.,0.),
                  spin=Vec3(0.,0.,0.))
    fast=airborne(pos=slow.position,vel=slow.velocity,
                  spin=Vec3(0.,6.,0.))
    a=prove(slow,duration=.1)
    b=prove(fast,duration=.1)
    assert a.valid and not b.valid
    assert b.angular_speed_bound_radps>a.angular_speed_bound_radps

def test_optin_enclosure_policy_is_separate_from_piecewise_ccd():
    sweep=TorqueFreeSweepPolicy(analytic_ode=POLICY)
    assert sweep.analytic_ode is POLICY
    with pytest.raises(ValueError,match="analytic_ode"):
        replace(sweep,analytic_ode="unsafe")

@pytest.mark.parametrize("field,value",[
    ("max_subdivisions",-1),("max_subdivisions",21),
    ("max_subdivisions",True),("min_clearance_m",float("nan")),
    ("min_clearance_m",0.),("min_clearance_m",.5),
])
def test_invalid_enclosure_policy_rejected(field,value):
    with pytest.raises(ValueError):
        replace(POLICY,**{field:value})


def test_world_flight_optin_applies_both_analytic_and_numerical_ccd_gates():
    from src.sandbox.ballistic_flight import (
        BallisticFlightPolicy,FlightPhysicsError,advance_ballistic_flight,
    )
    from src.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
    from src.sandbox.independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy
    from src.sandbox.contact_manifold import ContactManifoldPolicy

    contact=MultiSurfaceSuspensionPolicy(W,.12,.12)
    inertia=IndependentSuspensionDynamicsPolicy(
        contact,1200.,1800.,600.,30000.,2000.,150.,
        integration_substeps=8,
    )
    flight=BallisticFlightPolicy(
        contact,6.,.12,
        manifold=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.),
        torque_free=ANGULAR,
        torque_free_sweep=TorqueFreeSweepPolicy(
            slices=16,max_subdivisions=10,
            analytic_ode=POLICY,
        ),
    )
    safe=advance_ballistic_flight(
        airborne(),flight,BODY,(ROAD,),(),
        dt=.2,gravity_mps2=9.81,landing_dynamics=inertia,
    )
    assert safe.flight.phase=="airborne"
    assert safe.angular_audit is not None

    unsafe=airborne(
        pos=Vec3(0.,.3,0.),vel=Vec3(0.,0.,0.),
        spin=Vec3(0.,0.,0.),
    )
    with pytest.raises(FlightPhysicsError,match="analytic Euler ODE clearance unresolved"):
        advance_ballistic_flight(
            unsafe,flight,BODY,(ROAD,),(),
            dt=.05,gravity_mps2=9.81,landing_dynamics=inertia,
        )
