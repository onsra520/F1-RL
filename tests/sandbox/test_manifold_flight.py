"""Synthetic manifold touchdown/rebound integration for research-only WP7."""
from dataclasses import replace
import pytest
from f1rl.sandbox.contact_manifold import ContactManifoldPolicy
from f1rl.sandbox.ballistic_flight import (
    BallisticFlightPolicy, FlightPhysicsError, FlightState,
    advance_ballistic_flight,
)
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState,VehicleControl,VehicleParameters

BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
WHEELS=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
CONTACT=MultiSurfaceSuspensionPolicy(WHEELS,.12,.12)
PHYSICS=IndependentSuspensionDynamicsPolicy(
    CONTACT,1200.,1800.,600.,30000.,2000.,150.,integration_substeps=8,
)
MANIFOLD=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.)
FLIGHT=BallisticFlightPolicy(
    CONTACT,6.,.12,manifold=MANIFOLD,
)
ROAD=PlanarSurfacePatch(
    "flat",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,20.,
)


def source(*, angular=None,vy=-1.,y=.85):
    return VehicleState(
        Vec3(0.,y,0.),Vec3(0.,vy,0.),
        Quaternion.identity(),angular or Vec3(0.,0.,0.),False,
    )


def flight(start=None,policy=FLIGHT,dt=.35):
    return advance_ballistic_flight(
        start or source(),policy,BODY,(ROAD,),(),dt=dt,
        gravity_mps2=9.81,landing_dynamics=PHYSICS,
    )


def test_symmetric_four_wheel_landing_passes_exact_manifold_audit():
    out=flight()
    assert out.flight.phase=="grounded"
    assert out.vehicle.grounded
    assert out.manifold_audit is not None
    assert len(out.manifold_audit.contacts)==4
    assert all(x.normal_impulse_ns>=0. for x in out.manifold_audit.contacts)
    assert out.manifold_audit.post_kinetic_energy_j<=out.manifold_audit.pre_kinetic_energy_j
    assert out.landing_suspension is not None
    assert out.remaining_dt_s>0.
    assert [x.kind for x in out.transitions]==["landing"]


def test_bounce_continues_ballistic_to_full_tick_without_forged_ground():
    policy=replace(FLIGHT,manifold=replace(MANIFOLD,restitution=.5))
    out=flight(policy=policy)
    assert out.flight.phase=="airborne"
    assert not out.vehicle.grounded
    assert out.manifold_audit is not None
    assert out.manifold_audit.post_kinetic_energy_j<out.manifold_audit.pre_kinetic_energy_j
    assert out.remaining_dt_s==0.
    assert [x.kind for x in out.transitions]==["rebound"]
    assert out.vehicle.velocity.y>0.
    assert out.vehicle.position.y>.3


def test_no_impact_continues_without_fabricated_audit():
    out=flight(source(y=4.,vy=1.),dt=.05)
    assert not out.vehicle.grounded
    assert out.flight.phase=="airborne"
    assert out.manifold_audit is None
    assert out.transitions==()


def test_manifold_physics_requires_matching_chassis_registration():
    with pytest.raises(FlightPhysicsError,match="matching registered chassis inertia"):
        advance_ballistic_flight(
            source(),FLIGHT,BODY,(ROAD,),(),dt=.35,gravity_mps2=9.81,
        )


def test_manifold_and_old_impulse_policy_cannot_both_run():
    from f1rl.sandbox.angular_landing_impulse import AngularLandingImpulsePolicy
    with pytest.raises(ValueError,match="mutually exclusive"):
        replace(FLIGHT,angular_impulse=AngularLandingImpulsePolicy())


def test_rebound_cannot_jump_over_unresolved_second_contact():
    policy=replace(FLIGHT,manifold=replace(MANIFOLD,restitution=.02))
    with pytest.raises(FlightPhysicsError,match="rebound separation"):
        flight(policy=policy,dt=.4)


def test_world_manifold_grounded_handoff_maintains_phase_and_suspension():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.world import WorldState,step_world

    initial=source()
    state=WorldState(
        initial,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(initial.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    out=step_world(
        TrackWorld(surfaces=(ROAD,)),state,
        VehicleControl(0.,0.,0.),params,.35,.5,
        body_geometry=BODY,independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=FLIGHT,
    )
    assert out.state.vehicle.grounded
    assert out.state.flight.phase=="grounded"
    assert out.state.suspension is not None
    assert out.flight_telemetry.manifold_audit is not None
    assert [x.kind for x in out.events]==["landing"]


def test_world_rebound_does_not_issue_score_or_successful_grounding():
    from f1rl.sandbox.interactions.state import InteractionState
    from f1rl.sandbox.rules.scoring import ScoreState
    from f1rl.sandbox.track.model import TrackWorld
    from f1rl.sandbox.track.state import TrackState
    from f1rl.sandbox.world import WorldState,step_world

    initial=source()
    state=WorldState(
        initial,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(initial.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    policy=replace(FLIGHT,manifold=replace(MANIFOLD,restitution=.5))
    out=step_world(
        TrackWorld(surfaces=(ROAD,)),state,
        VehicleControl(0.,0.,0.),params,.35,.5,
        body_geometry=BODY,independent_suspension_policy=PHYSICS,
        ballistic_flight_policy=policy,
    )
    assert not out.state.vehicle.grounded
    assert out.state.flight.phase=="airborne"
    assert out.state.score==state.score
    assert [x.kind for x in out.events]==["rebound"]
