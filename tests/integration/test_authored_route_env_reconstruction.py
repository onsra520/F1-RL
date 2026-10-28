"""C2.2b: optional authored route guidance is wired through env and factory."""
from pathlib import Path

import numpy as np
import pytest

from src.rl import EpisodeConfig, NavigationTarget, ObservationConfig, ObservationEncoder, RewardComposer
from src.rl.episode import RaceProgressState
from src.rl.gym_environment import F1RLEnv
from src.training.env_factory import F1RLEnvFactory
from src.core.simulation import SimulationConfig
from src.simulation.geometry.vector import Vec3
from src.simulation.perception.sensor_suite import SensorSuite
from src.simulation.vehicle import VehicleParameters
from src.tracks import build_initial_world_state, load_track_asset


def _factory(track_id: str, *, guided: bool) -> F1RLEnvFactory:
    asset=load_track_asset(Path("assets/tracks/training") / f"{track_id}.json")
    return F1RLEnvFactory(
        track=asset.world,
        initial_state=build_initial_world_state(asset,max_surface_distance=0.75),
        vehicle_parameters=VehicleParameters(3.0,0.5,0.5,6.0,10.0,0.4,0.1,15.0,9.81),
        simulation_config=SimulationConfig(0.1,0.75),
        observation_encoder=ObservationEncoder(ObservationConfig(SensorSuite(()),15.0,80.0)),
        reward_composer=RewardComposer(()),
        episode_config=EpisodeConfig(500),
        navigation=asset.navigation if guided else None,
    )


def test_opt_in_guidance_targets_next_authored_endpoint_not_distant_checkpoint() -> None:
    guided=_factory("ramp-v1",guided=True).create()
    legacy=_factory("ramp-v1",guided=False).create()
    a,_=guided.reset(seed=11)
    b,_=legacy.reset(seed=11)
    assert guided.observation_space.shape == legacy.observation_space.shape
    assert not np.array_equal(a,b)
    vehicle=guided._simulation.snapshot.world.vehicle
    expected=guided._observation_encoder.encode(
        guided._track,vehicle,NavigationTarget(Vec3(0,0,40))
    )
    np.testing.assert_allclose(a,np.asarray(expected,dtype=np.float32))
    assert guided._route_target(vehicle,RaceProgressState(0,False)) == Vec3(0,0,40)
    assert legacy._route_target(vehicle,RaceProgressState(0,False)) == Vec3(0,3.479327524001636,74.75670171853926)


def test_corner_guidance_uses_polyline_segment_for_second_checkpoint() -> None:
    guided=_factory("right-turn-v1",guided=True).create()
    legacy=_factory("right-turn-v1",guided=False).create()
    guided.reset()
    legacy.reset()
    checkpoint_state=RaceProgressState(1,False)
    position=guided._simulation.snapshot.world.vehicle
    moved=type(position)(
        position=Vec3(0,0,65),
        velocity=position.velocity,
        orientation=position.orientation,
        angular_velocity=position.angular_velocity,
        grounded=position.grounded,
    )
    assert guided._route_target(moved,checkpoint_state) == Vec3(0,0,70)
    assert legacy._route_target(moved,checkpoint_state) == Vec3(20,0,70)


def test_route_guidance_step_observation_matches_runtime_vehicle_projection() -> None:
    env=_factory("ramp-v1",guided=True).create()
    env.reset()
    obs,_,_,_,_=env.step(np.array([0.0,0.0],dtype=np.float32))
    vehicle=env._simulation.snapshot.world.vehicle
    target=env._route_target(vehicle,env._progress)
    expected=env._observation_encoder.encode(env._track,vehicle,NavigationTarget(target))
    np.testing.assert_allclose(obs,np.asarray(expected,dtype=np.float32))


def test_guided_factory_rejects_bad_navigation_type() -> None:
    factory=_factory("ramp-v1",guided=False)
    from dataclasses import replace
    with pytest.raises(ValueError,match="navigation"):
        replace(factory,navigation=object())


def test_guided_route_validates_track_consistency() -> None:
    from dataclasses import replace
    first=_factory("straight-sprint-v1",guided=False)
    second=_factory("ramp-v1",guided=True)
    with pytest.raises(ValueError,match="navigation"):
        replace(first,navigation=second.navigation).create()


def test_finished_route_has_no_navigation_target() -> None:
    env=_factory("straight-sprint-v1",guided=True).create()
    env.reset()
    position=env._simulation.snapshot.world.vehicle
    assert env._route_target(position,RaceProgressState(2,True)) is None
