"""Synthetic physics integration contracts, never empirical calibration."""
import pytest
import numpy as np
from dataclasses import replace

from tests.learning.training.mocks.mock_factories import make_factory

from f1rl.runtime.physics import PhysicsConfig
from f1rl.runtime.simulation import Simulation, SimulationConfig
from f1rl.sandbox.body_collision import VehicleBodyGeometry
from f1rl.sandbox.ballistic_flight import BallisticFlightPolicy, FlightState
from f1rl.sandbox.grounded_yaw_dynamics import GroundedYawPolicy, GroundedYawResolutionError
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.multi_surface_support import MultiSurfaceSuspensionPolicy
from f1rl.sandbox.ramp_support import WheelContactPoints
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


BODY = VehicleBodyGeometry(1.8, 1.2, 4., Vec3(0., .3, 0.))
WHEELS = WheelContactPoints(
    Vec3(-.75, -.5, 1.5), Vec3(.75, -.5, 1.5),
    Vec3(-.75, -.5, -1.5), Vec3(.75, -.5, -1.5),
)
CONTACT = MultiSurfaceSuspensionPolicy(WHEELS, .12, .12)
SUSPENSION = IndependentSuspensionDynamicsPolicy(
    CONTACT, 1200., 1800., 600., 30000., 2000., 150.,
)
PARAMETERS = VehicleParameters(3., .5, .6, 0., 20., 0., 0., 50., 9.81)
ROAD = PlanarSurfacePatch(
    "road", Vec3(0., 0., 0.), Vec3(0., 0., 1.), Vec3(0., 1., 0.), 20., 100.,
)


def world(*, height=.5, speed=5., vertical_speed=0., grounded=True, flight=None):
    return WorldState(
        VehicleState(Vec3(0., height, 0.), Vec3(0., vertical_speed, speed),
                     Quaternion.identity(), Vec3(0., 0., 0.), grounded),
        TrackState(frozenset()), InteractionState(frozenset(), frozenset()),
        ScoreState(), suspension=SuspensionDynamicsState(height), flight=flight,
    )


def simulation(config, initial=None, *, track=None, dt=.05):
    return Simulation(
        track or TrackWorld(surfaces=(ROAD,)), initial or world(),
        PARAMETERS, SimulationConfig(dt, 1.), physics_config=config,
    )


def test_registered_suspension_reaches_runtime_and_persists_physical_state():
    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    initial = world(height=.52)
    sim = simulation(config, initial)
    result = sim.step(VehicleControl(0., 0., 0.))
    assert .5 < result.world.vehicle.position.y < .52
    assert result.world.suspension.height_m == result.world.vehicle.position.y
    assert result.suspension_telemetry is not None
    assert len(result.suspension_telemetry.force_samples[-1]) == 4
    assert result.tick == 1
    assert result.elapsed_time == pytest.approx(.05)
    assert sim.reset().world == initial


def factory(config, initial=None):
    return replace(
        make_factory(), initial_state=initial or world(height=.52),
        vehicle_parameters=PARAMETERS, simulation_config=SimulationConfig(.05, 1.),
        physics_config=config,
    )


def test_factory_gym_exposes_registered_config_and_actual_suspension_response():
    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    env = factory(config).create()
    try:
        observation, info = env.reset(seed=17)
        assert observation.shape == (9,)
        assert info["physics"]["sha256"] == config.sha256
        assert info["physics"]["configuration"] == config.document()
        _, _, terminated, truncated, info = env.step(np.array([0., 0.], dtype=np.float32))
        assert .5 < info["telemetry"]["vehicle_position"][1] < .52
        assert info["physics"]["suspension"] is not None
        assert not terminated and not truncated
        assert env.reset(seed=17)[1]["physics"]["suspension"] is None
    finally:
        env.close()


@pytest.mark.parametrize("multi", [False, True])
def test_new_physics_cannot_bypass_runner_admission(tmp_path, monkeypatch, multi):
    from f1rl.learning.algorithms.ppo import PPOConfig
    from f1rl.learning.training.config import TrainingRunConfig, VectorEnvConfig
    from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
    from f1rl.learning.training.runner import TrainingRunner

    def forbidden_run(*args, **kwargs):
        raise AssertionError("training reached before admission")

    monkeypatch.setattr(TrainingRunner, "_run", forbidden_run)
    candidate = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION))
    root = tmp_path / "forbidden-artifacts"
    runner = TrainingRunner(TrainingRunConfig(
        PPOConfig(seed=1), VectorEnvConfig(n_envs=1), artifacts_root=root,
    ))
    with pytest.raises(ValueError, match="full-body candidate training is blocked"):
        if multi:
            runner.run_multi(TrainingScenarioSet((TrainingScenario("synthetic", candidate),)))
        else:
            runner.run_single(candidate)
    assert not root.exists()


def test_dynamic_yaw_reaches_gym_and_low_speed_never_falls_back():
    yaw = GroundedYawPolicy(1200., 1600., 1.5, 1.5, 18000., 18000., .8)
    config = PhysicsConfig("wp7_grounded_yaw_v1", BODY, SUSPENSION, grounded_yaw_policy=yaw)
    env = factory(config).create()
    try:
        env.reset()
        info = env.step(np.array([.2, 0.], dtype=np.float32))[-1]
        assert info["physics"]["grounded_yaw"]["yaw_moment_nm"] > 0.
        assert info["telemetry"]["vehicle_position"][0] > 0.
    finally:
        env.close()
    sim = simulation(config, world(speed=.1))
    before = sim.snapshot
    with pytest.raises(GroundedYawResolutionError, match="low speed"):
        sim.step(VehicleControl(.2, 0., 0.))
    assert sim.snapshot is before


def test_ballistic_configuration_reaches_gym_with_exact_gravity():
    flight = BallisticFlightPolicy(CONTACT, 3., .1)
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    initial = world(height=5., vertical_speed=1., grounded=False, flight=FlightState("airborne"))
    env = factory(config, initial).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        assert info["telemetry"]["vehicle_position"][1] == pytest.approx(5.0377375)
        assert info["telemetry"]["vehicle_velocity"][1] == pytest.approx(.5095)
        assert info["physics"]["flight"]["flight"]["phase"] == "airborne"
    finally:
        env.close()


def test_legacy_contract_preserves_baseline_state_and_policy_spaces():
    baseline = make_factory()
    explicit = replace(baseline, physics_config=PhysicsConfig("legacy_v1"))
    old, new = baseline.create(), explicit.create()
    try:
        assert old.action_space == new.action_space
        assert old.observation_space == new.observation_space
        assert np.array_equal(old.reset(seed=7)[0], new.reset(seed=7)[0])
        for _ in range(3):
            action = np.array([.1, .4], dtype=np.float32)
            a, b = old.step(action), new.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            assert a[4] == {k: v for k, v in b[4].items() if k != "physics"}
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"},
    {"mode": "legacy_v1", "body_geometry": BODY},
    {"mode": "wp7_suspension_v1", "body_geometry": BODY},
    {"mode": "wp7_suspension_v1", "body_geometry": BODY,
     "independent_suspension_policy": SUSPENSION, "contact_geometry": "tire_hubs_v1"},
    {"mode": "wp7_ballistic_v1", "body_geometry": BODY,
     "independent_suspension_policy": SUSPENSION},
    {"mode": "wp7_grounded_yaw_v1", "body_geometry": BODY,
     "independent_suspension_policy": SUSPENSION},
])
def test_unregistered_or_incomplete_physics_cannot_construct(kwargs):
    with pytest.raises(ValueError):
        PhysicsConfig(**kwargs)


def test_owner_geometry_and_legacy_options_cannot_override_registration():
    with pytest.raises(ValueError, match="owner body"):
        PhysicsConfig("wp7_suspension_v1", replace(BODY, width_m=2.), SUSPENSION)
    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    with pytest.raises(ValueError, match="cannot be mixed"):
        replace(factory(config), body_geometry=BODY)
    with pytest.raises(ValueError, match="persistent suspension"):
        factory(config, replace(world(), suspension=None))
    with pytest.raises(ValueError, match="explicit WP7"):
        simulation(None)


def test_scenario_contract_and_vector_slots_preserve_physics_identity():
    from f1rl.learning.training.env_factory import TrainingScenario, TrainingScenarioSet
    from f1rl.learning.training.vector_env import make_multi_track_vector_env
    from f1rl.learning.training.config import VectorEnvConfig

    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    a = TrainingScenario("synthetic-a", factory(config))
    b = TrainingScenario("synthetic-b", factory(config))
    changed = replace(config, independent_suspension_policy=replace(SUSPENSION, damper_nspm=2500.))
    assert config.sha256 != changed.sha256
    with pytest.raises(ValueError, match="non-track environment contract"):
        TrainingScenarioSet((a, TrainingScenario(b.track_id, factory(changed))))
    vec = make_multi_track_vector_env(TrainingScenarioSet((a, b)), VectorEnvConfig(n_envs=2), 13)
    try:
        assert vec.reset().shape == (2, 9)
        _, _, _, infos = vec.step(np.zeros((2, 2), dtype=np.float32))
        assert all(info["physics"]["sha256"] == config.sha256 for info in infos)
        assert all(info["physics"]["suspension"] is not None for info in infos)
    finally:
        vec.close()
