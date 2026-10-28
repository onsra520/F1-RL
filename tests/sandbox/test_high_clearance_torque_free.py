"""6DOF torque-free only in certified obstacle-free high-altitude flight."""
from dataclasses import replace
import pytest

from src.simulation.ballistic_flight import (
    BallisticFlightPolicy, FlightState, FlightPhysicsError,
    advance_ballistic_flight,
)
from src.simulation.contact_manifold import ContactManifoldPolicy
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.geometry.vector import Vec3
from src.simulation.geometry.rotation import Quaternion
from src.simulation.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from src.simulation.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.simulation.ramp_support import WheelContactPoints
from src.simulation.torque_free_rotation import TorqueFreeRotationPolicy
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.volume import OrientedBoxVolume
from src.simulation.vehicle import VehicleState,VehicleControl,VehicleParameters

BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
CONTACT=MultiSurfaceSuspensionPolicy(W,.12,.12)
PHYS=IndependentSuspensionDynamicsPolicy(
    CONTACT,1200.,1800.,600.,30000.,2000.,150.,integration_substeps=8,
)
MANIFOLD=ContactManifoldPolicy(yaw_inertia_kg_m2=1600.)
FREE=TorqueFreeRotationPolicy(
    1800.,1600.,600.,integration_steps=64,
)
FLIGHT=BallisticFlightPolicy(
    CONTACT,6.,.12,manifold=MANIFOLD,torque_free=FREE,
)
ROAD=PlanarSurfacePatch(
    "flat",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),10.,20.,
)

def air(y=5.,w=Vec3(.15,.24,.17)):
    return VehicleState(
        Vec3(0.,y,0.),Vec3(1.,1.,2.),
        Quaternion.identity(),w,False,
    )

def fly(start=None,policy=FLIGHT,obstacles=(),dt=.2):
    return advance_ballistic_flight(
        start or air(),policy,BODY,(ROAD,),obstacles,
        dt=dt,gravity_mps2=9.81,landing_dynamics=PHYS,
    )


def test_high_clearance_torque_free_is_advanced_and_audited():
    original=air()
    out=fly(original)
    assert out.flight.phase=="airborne"
    assert not out.vehicle.grounded
    assert out.angular_audit is not None
    assert out.angular_audit.final_energy_j==pytest.approx(
        out.angular_audit.initial_energy_j,rel=1e-7,
    )
    assert (out.angular_audit.final_world_angular_momentum
            -out.angular_audit.initial_world_angular_momentum).magnitude<1e-5
    assert out.vehicle.orientation!=original.orientation
    assert out.vehicle.angular_velocity!=original.angular_velocity
    assert out.vehicle.position.y==pytest.approx(5.+.2-.5*9.81*.2*.2)
    assert out.vehicle.position.x==pytest.approx(.2)
    assert out.vehicle.position.z==pytest.approx(.4)
    assert out.transitions==()


def test_low_altitude_near_road_refused_even_if_current_pose_clear():
    with pytest.raises(FlightPhysicsError,match="road clearance not globally certified"):
        fly(air(y=1.1))


def test_obstacle_sweep_not_assumed_safe_under_non_slerp_precession():
    far=OrientedBoxVolume(
        Vec3(0.,8.,50.),Vec3(1.,1.,1.),Quaternion.identity(),
    )
    with pytest.raises(FlightPhysicsError,match="collision sweep not yet certified"):
        fly(obstacles=(far,))


def test_inertia_must_equal_ground_body_and_registered_manifold():
    mismatch=replace(FLIGHT,torque_free=replace(FREE,yaw_inertia_kg_m2=1700.))
    with pytest.raises(FlightPhysicsError,match="inertia registrations mismatch"):
        fly(policy=mismatch)


def test_optin_requires_explicit_manifold_and_ground_inertia():
    policy=replace(FLIGHT,manifold=None)
    with pytest.raises(FlightPhysicsError,match="requires registered 3-axis inertia"):
        fly(policy=policy)
    with pytest.raises(FlightPhysicsError,match="requires registered 3-axis inertia"):
        advance_ballistic_flight(
            air(),FLIGHT,BODY,(ROAD,),(),
            dt=.2,gravity_mps2=9.81,
        )


def test_world_preserves_torque_free_angular_state_without_earning_events():
    from src.simulation.interactions.state import InteractionState
    from src.simulation.rules.scoring import ScoreState
    from src.simulation.track.model import TrackWorld
    from src.simulation.track.state import TrackState
    from src.simulation.world import WorldState,step_world
    initial=air()
    state=WorldState(
        initial,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(initial.position.y),
        flight=FlightState("airborne"),
    )
    params=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)
    out=step_world(
        TrackWorld(surfaces=(ROAD,)),state,
        VehicleControl(0.,0.,0.),params,.2,.5,
        body_geometry=BODY,independent_suspension_policy=PHYS,
        ballistic_flight_policy=FLIGHT,
    )
    assert out.state.flight.phase=="airborne"
    assert not out.state.vehicle.grounded
    assert out.flight_telemetry.angular_audit is not None
    assert out.state.vehicle.angular_velocity!=initial.angular_velocity
    assert out.state.score==state.score
    assert out.events==()
