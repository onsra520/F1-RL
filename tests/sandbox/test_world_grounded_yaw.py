"""Opt-in dynamic yaw through World -> independent suspension with full CCD."""
import pytest

from f1rl.sandbox.grounded_yaw_dynamics import GroundedYawPolicy
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy,SuspensionDynamicsState,
)
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.vehicle import VehicleState,VehicleControl,VehicleParameters
from f1rl.sandbox.world import WorldState,step_world

W=WheelContactPoints(
    Vec3(-.4,-.3,.8),Vec3(.4,-.3,.8),
    Vec3(-.4,-.3,-.8),Vec3(.4,-.3,-.8),
)
BODY=VehicleBodyGeometry(1.,.4,2.,Vec3(0.,.1,0.))
CONTACT=MultiSurfaceSuspensionPolicy(W,.12,.12)
SUSP=IndependentSuspensionDynamicsPolicy(
    CONTACT,1200.,1800.,600.,30000.,2000.,150.,integration_substeps=8,
)
YAW=GroundedYawPolicy(
    1200.,1600.,1.25,1.25,18000.,18000.,.8,
    min_longitudinal_speed_mps=.5,
)
ROAD=PlanarSurfacePatch(
    "flat",Vec3(0.,0.,0.),Vec3(0.,0.,1.),Vec3(0.,1.,0.),20.,30.,
)
PARAM=VehicleParameters(2.5,.3,.6,0.,20.,0.,0.,50.,9.81)


def scene(speed=5.,spin=0.):
    vehicle=VehicleState(
        Vec3(0.,.3,0.),Vec3(0.,0.,speed),Quaternion.identity(),
        Vec3(0.,spin,0.),True,
    )
    return WorldState(
        vehicle,TrackState(frozenset()),
        InteractionState(frozenset(),frozenset()),ScoreState(),
        suspension=SuspensionDynamicsState(.3),
    )


def step(initial,steering=.2,yaw=YAW):
    return step_world(
        TrackWorld(surfaces=(ROAD,)),initial,
        VehicleControl(steering,0.,0.),PARAM,.05,.5,
        body_geometry=BODY,independent_suspension_policy=SUSP,
        grounded_yaw_policy=yaw,
    )


def test_grounded_yaw_world_advances_suspension_without_phantom_score():
    initial=scene()
    out=step(initial)
    assert out.state.vehicle.grounded
    assert out.state.vehicle.angular_velocity.y>0.
    assert out.state.vehicle.position.x>0.
    assert out.state.suspension is not None
    assert out.grounded_yaw_telemetry is not None
    assert out.grounded_yaw_telemetry.front_force_n>0.
    assert out.state.score==initial.score
    assert out.events==()


def test_existing_yaw_rate_is_carried_through_world_step():
    initial=scene(spin=.1)
    out=step(initial,steering=0.)
    assert out.state.vehicle.angular_velocity.y!=0.
    assert out.grounded_yaw_telemetry is not None


def test_invalid_yaw_mass_and_axle_geometry_block_world():
    from dataclasses import replace
    with pytest.raises(ValueError,match="mass must match"):
        step(scene(),yaw=replace(YAW,mass_kg=1300.))
    with pytest.raises(ValueError,match="axle geometry"):
        step(scene(),yaw=replace(YAW,front_axle_distance_m=1.4))


def test_low_speed_does_not_silently_fall_back_to_legacy_yaw():
    from f1rl.sandbox.grounded_yaw_dynamics import GroundedYawResolutionError
    with pytest.raises(GroundedYawResolutionError,match="low speed"):
        step(scene(speed=.1))


def test_optin_yaw_is_deterministic():
    assert step(scene())==step(scene())
