"""WP7 research-only inclined manifold and time-ordered bouncing cases."""
from dataclasses import replace
from math import sin,cos
import pytest

from src.simulation.ballistic_flight import (
    BallisticFlightPolicy, FlightState, FlightPhysicsError,
    advance_ballistic_flight, _flight_pose,
)
from src.simulation.contact_manifold import ContactManifoldPolicy
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.independent_suspension_dynamics import IndependentSuspensionDynamicsPolicy
from src.simulation.multi_surface_support import MultiSurfaceSuspensionPolicy, _atlas
from src.simulation.ramp_support import WheelContactPoints
from src.simulation.rotating_touchdown import first_rotating_wheel_touchdown
from src.simulation.angular_landing_impulse import AngularLandingImpulsePolicy
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.vehicle import VehicleState

W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
CONTACT=MultiSurfaceSuspensionPolicy(W,.12,.12)
PHYS=IndependentSuspensionDynamicsPolicy(CONTACT,1200.,1800.,600.,30000.,2000.,150.,
                                          integration_substeps=8)
BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
MANIFOLD=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.)
FLIGHT=BallisticFlightPolicy(CONTACT,6.,.12,manifold=MANIFOLD)


def patch(angle=0.):
    return PlanarSurfacePatch(
        "incline",Vec3(0.,0.,0.),Vec3(0.,sin(angle),cos(angle)),
        Vec3(0.,cos(angle),-sin(angle)),10.,20.,
    )


def state(*,y=.85,vy=-1.,pitch=0.,wx=0.,vz=0.):
    return VehicleState(
        Vec3(0.,y,0.),Vec3(0.,vy,vz),
        Quaternion.from_axis_angle(Vec3(1.,0.,0.),pitch),
        Vec3(wx,0.,0.),False,
    )


def simulate(*,initial=None,dt=.35,angle=0.,policy=FLIGHT):
    return advance_ballistic_flight(
        initial or state(),policy,BODY,(patch(angle),),(),
        dt=dt,gravity_mps2=9.81,landing_dynamics=PHYS,
    )


def test_declared_small_slope_symmetric_manifold_touchdown():
    angle=.018
    policy=replace(FLIGHT,max_patch_tilt_rad=.03)
    out=simulate(initial=state(pitch=-angle),angle=angle,policy=policy)
    assert out.vehicle.grounded
    assert out.flight.phase=="grounded"
    assert out.manifold_audit is not None
    assert len(out.manifold_audit.contacts)==4
    assert out.manifold_audit.post_kinetic_energy_j<=out.manifold_audit.pre_kinetic_energy_j
    assert out.transitions[0].kind=="landing"


def test_slope_outside_registered_acceptance_still_fails_closed():
    with pytest.raises(FlightPhysicsError,match="sloped touchdown"):
        simulate(initial=state(pitch=-.018),angle=.018)


def test_sloped_first_contact_is_localized_against_true_plane_normal():
    angle=.025
    surface=patch(angle)
    atlas=_atlas((surface,),CONTACT.seam_height_tolerance_m)
    initial=state(pitch=-angle,wx=.01)
    first=first_rotating_wheel_touchdown(
        lambda t:_flight_pose(initial,t,9.81),
        W,atlas,.35,9.81,AngularLandingImpulsePolicy(),
    )
    assert first is not None
    assert 0.<first.time_s<.35
    assert 0<=first.wheel_index<4


def test_two_rebounds_inside_one_tick_have_ordered_timestamps_and_audits():
    policy=replace(FLIGHT,manifold=replace(
        MANIFOLD,restitution=.5,max_impacts_per_step=2,
    ))
    out=simulate(dt=.67,policy=policy)
    assert not out.vehicle.grounded
    assert out.flight.phase=="airborne"
    assert len(out.transitions)==2
    assert [e.kind for e in out.transitions]==["rebound","rebound"]
    assert 0.<out.transitions[0].time_s<out.transitions[1].time_s<.67
    assert len(out.manifold_history)==2
    assert all(a.post_kinetic_energy_j<a.pre_kinetic_energy_j for a in out.manifold_history)
    assert out.remaining_dt_s==0.


def test_old_one_event_budget_refuses_uncertified_second_contact():
    policy=replace(FLIGHT,manifold=replace(
        MANIFOLD,restitution=.5,max_impacts_per_step=1,
    ))
    with pytest.raises(FlightPhysicsError,match="rebound separation"):
        simulate(dt=.67,policy=policy)


@pytest.mark.parametrize("budget",[0,9,1.5,True])
def test_invalid_impact_budget_fails_before_simulation(budget):
    with pytest.raises(ValueError,match="max_impacts_per_step"):
        replace(MANIFOLD,max_impacts_per_step=budget)


def test_replay_of_multiple_impacts_is_deterministic():
    policy=replace(FLIGHT,manifold=replace(
        MANIFOLD,restitution=.5,max_impacts_per_step=2,
    ))
    assert simulate(dt=.67,policy=policy)==simulate(dt=.67,policy=policy)


def test_first_impact_bracket_is_nonpenetrating_and_bounded():
    surface=patch()
    atlas=_atlas((surface,),CONTACT.seam_height_tolerance_m)
    initial=state()
    option=AngularLandingImpulsePolicy()
    first=first_rotating_wheel_touchdown(
        lambda t:_flight_pose(initial,t,9.81),
        W,atlas,.35,9.81,option,
    )
    assert first is not None
    impact=_flight_pose(initial,first.time_s,9.81)
    point=impact.position+impact.orientation.rotate(W.positions[first.wheel_index])
    signed_gap=(point-surface.origin).dot(surface.normal)
    # Never produce the already-penetrating side of a time bracket.
    assert signed_gap >= -1e-10
    assert signed_gap <= 1e-6
