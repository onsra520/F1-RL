"""Synthetic physics integration contracts, never empirical calibration."""
import pytest
import numpy as np
from dataclasses import replace

from tests.learning.training.mocks.mock_factories import make_factory

from src.core.physics import PhysicsConfig
from src.core.simulation import Simulation, SimulationConfig
from src.simulation.body_collision import VehicleBodyGeometry
from src.simulation.ballistic_flight import BallisticFlightPolicy, FlightState
from src.simulation.grounded_yaw_dynamics import GroundedYawPolicy, GroundedYawResolutionError
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.independent_suspension_dynamics import (
    IndependentSuspensionDynamicsPolicy, SuspensionDynamicsState,
)
from src.simulation.interactions.state import InteractionState
from src.simulation.multi_surface_support import MultiSurfaceSuspensionPolicy
from src.simulation.ramp_support import WheelContactPoints
from src.simulation.rules.scoring import ScoreState
from src.simulation.track.model import TrackWorld
from src.simulation.track.patch import PlanarSurfacePatch
from src.simulation.track.state import TrackState
from src.simulation.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.simulation.world import WorldState


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
    from src.rl.algorithms.ppo import PPOConfig
    from src.training.config import TrainingRunConfig, VectorEnvConfig
    from src.training.env_factory import TrainingScenario, TrainingScenarioSet
    from src.training.runner import TrainingRunner

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
    from src.training.env_factory import TrainingScenario, TrainingScenarioSet
    from src.training.vector_env import make_multi_track_vector_env
    from src.training.config import VectorEnvConfig

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


def test_research_ground_collision_rejects_unregistered_rollback_recovery():
    from src.simulation.entities.obstacle import Obstacle
    from src.simulation.track.volume import OrientedBoxVolume

    obstacle = Obstacle("wall", OrientedBoxVolume(
        Vec3(0., .8, 2.2), Vec3(.5, .8, .02), Quaternion.identity(),
    ))
    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    sim = simulation(config, track=TrackWorld(surfaces=(ROAD,), obstacles=(obstacle,)))
    before = sim.snapshot
    with pytest.raises(RuntimeError, match="collision recovery is unregistered"):
        sim.step(VehicleControl(0., 0., 0.))
    assert sim.snapshot is before


def test_airborne_snapshot_can_reconstruct_with_last_grounded_suspension_state():
    flight = BallisticFlightPolicy(CONTACT, 3., .1)
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    initial = replace(
        world(height=5., grounded=False, flight=FlightState("airborne")),
        suspension=SuspensionDynamicsState(.5),
    )
    sim = simulation(config, initial)
    result = sim.step(VehicleControl(0., 0., 0.))
    rebuilt = simulation(config, result.world)
    assert rebuilt.step(VehicleControl(0., 0., 0.)).world == sim.step(VehicleControl(0., 0., 0.)).world


@pytest.mark.parametrize("restitution", [0., .5])
def test_gym_manifold_landing_and_rebound_preserve_event_time_and_energy(restitution):
    from math import sqrt
    from src.simulation.contact_manifold import ContactManifoldPolicy

    flight = BallisticFlightPolicy(
        CONTACT, 6., .12,
        manifold=ContactManifoldPolicy(1600., restitution=restitution),
    )
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    initial = world(height=.85, speed=0., vertical_speed=-1., grounded=False,
                    flight=FlightState("airborne"))
    env = replace(factory(config, initial), simulation_config=SimulationConfig(.25, 1.)).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        telemetry = info["physics"]["flight"]
        event = telemetry["transitions"][0]
        expected_time = (sqrt(1. + 2.*9.81*.35) - 1.) / 9.81
        assert event["time_s"] == pytest.approx(expected_time, abs=1e-7)
        assert info["elapsed_time"] == .25
        assert info["next_checkpoint_index"] == 0
        assert not info["finished"]
        assert info["game_score"] == 0
        assert info["reward_components"] == {}
        audit = telemetry["manifold_audit"]
        assert len(audit["contacts"]) == 4
        assert audit["pre_kinetic_energy_j"] == pytest.approx(4720.2, abs=.01)
        assert audit["post_kinetic_energy_j"] <= audit["pre_kinetic_energy_j"] + 1e-5
        assert audit["post_kinetic_energy_j"] == pytest.approx(4720.2*restitution**2, abs=.1)
        assert info["telemetry"]["grounded"] is (restitution == 0.)
        assert event["kind"] == ("landing" if restitution == 0. else "rebound")
        if restitution == 0.:
            assert telemetry["remaining_dt_s"] == pytest.approx(.25-expected_time)
            assert info["physics"]["suspension"] is not None
        else:
            assert telemetry["remaining_dt_s"] == 0.
            assert info["telemetry"]["vehicle_velocity"][1] > 0.
    finally:
        env.close()


def test_registered_ode_clearance_reaches_runtime_and_never_forges_contact():
    from src.simulation.ballistic_flight import FlightPhysicsError
    from src.simulation.contact_manifold import ContactManifoldPolicy
    from src.simulation.euler_ode_clearance import EulerODEEnclosurePolicy
    from src.simulation.torque_free_rotation import TorqueFreeRotationPolicy
    from src.simulation.torque_free_swept_ccd import TorqueFreeSweepPolicy

    flight = BallisticFlightPolicy(
        CONTACT, 6., .12, manifold=ContactManifoldPolicy(1600.),
        torque_free=TorqueFreeRotationPolicy(1800., 1600., 600.),
        torque_free_sweep=TorqueFreeSweepPolicy(analytic_ode=EulerODEEnclosurePolicy()),
    )
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    start = world(height=5., speed=0., grounded=False, flight=FlightState("airborne"))
    start = replace(start, vehicle=replace(start.vehicle, angular_velocity=Vec3(0., .7, 0.)))
    safe = simulation(config, start).step(VehicleControl(0., 0., 0.))
    assert safe.flight_telemetry.angular_audit is not None
    assert safe.events == ()
    assert safe.world.vehicle.position.y == pytest.approx(4.9877375)
    near = simulation(config, world(height=.5, speed=0., grounded=False,
                                   flight=FlightState("airborne")))
    before = near.snapshot
    with pytest.raises(FlightPhysicsError, match="analytic Euler ODE clearance unresolved"):
        near.step(VehicleControl(0., 0., 0.))
    assert near.snapshot is before


def test_grounded_seam_is_supported_but_mixed_patch_landing_is_rejected():
    from src.simulation.ballistic_flight import FlightPhysicsError
    from src.simulation.contact_manifold import ContactManifoldPolicy

    surfaces = (
        replace(ROAD, id="left", origin=Vec3(-5., 0., 0.), width=10.),
        replace(ROAD, id="right", origin=Vec3(5., 0., 0.), width=10.),
    )
    track = TrackWorld(surfaces=surfaces)
    grounded = simulation(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION), track=track)
    out = grounded.step(VehicleControl(0., 0., 0.))
    assert {s.contact.patch_id for s in out.suspension_telemetry.force_samples[-1]} == {"left", "right"}
    flight = BallisticFlightPolicy(CONTACT, 6., .12, manifold=ContactManifoldPolicy(1600.))
    initial = world(height=.85, speed=0., vertical_speed=-1., grounded=False,
                    flight=FlightState("airborne"))
    airborne = simulation(PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight),
                          initial, track=track, dt=.25)
    before = airborne.snapshot
    with pytest.raises(FlightPhysicsError, match="multi-surface simultaneous contact unsupported"):
        airborne.step(VehicleControl(0., 0., 0.))
    assert airborne.snapshot is before


def test_registered_automatic_edge_release_reaches_gym_without_phantom_race_events():
    from src.simulation.partial_wheel_release import AutomaticTakeoffPolicy
    flight = BallisticFlightPolicy(CONTACT, 6., .15)
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight,
                           automatic_takeoff_policy=AutomaticTakeoffPolicy(.6))
    candidate = factory(config, world(speed=40., flight=FlightState("grounded")))
    track = replace(candidate.track, surfaces=(replace(ROAD, length=4.),))
    env = replace(candidate, track=track, simulation_config=SimulationConfig(.1, 1.)).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        assert not info["telemetry"]["grounded"]
        partial = info["physics"]["partial_support"]
        assert 0. < partial["takeoff_at_s"] < .1
        assert any(sum(c is not None for c in frame["contacts"]) == 2 for frame in partial["frames"])
        assert not info["finished"]
        assert info["game_score"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("velocity,omega", [
    (Vec3(0., 2., 5.), Vec3(0., 0., 0.)),
    (Vec3(0., 0., 5.), Vec3(1., 0., 0.)),
    (Vec3(0., 0., 5.), Vec3(0., 0., 1.)),
])
def test_inconsistent_grounded_rates_are_rejected_before_admission_or_reset(velocity, omega):
    config = PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION)
    initial = world()
    invalid = replace(initial, vehicle=replace(initial.vehicle, velocity=velocity, angular_velocity=omega))
    with pytest.raises(ValueError, match="suspension rates disagree"):
        simulation(config, invalid)
    sim = simulation(config)
    before = sim.snapshot
    with pytest.raises(ValueError, match="suspension rates disagree"):
        sim.reset(initial_state=invalid)
    assert sim.snapshot is before


def test_grounded_rate_consistency_uses_world_frame_and_allows_consistent_rates():
    from math import pi
    initial = world()
    initial = replace(initial,
        vehicle=replace(initial.vehicle,
            orientation=Quaternion.from_axis_angle(Vec3(0., 1., 0.), pi/2),
            velocity=Vec3(5., .05, 0.), angular_velocity=Vec3(.2, 0., -.1)),
        suspension=replace(initial.suspension, heave_velocity_mps=.05, pitch_rate_radps=.1, roll_rate_radps=.2))
    sim = simulation(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION), initial)
    assert sim.snapshot.world == initial
    assert sim.step(VehicleControl(0., 0., 0.)).suspension_telemetry is not None


def test_explicit_takeoff_keeps_its_separate_authorized_upward_release_domain():
    flight = BallisticFlightPolicy(CONTACT, 3., .1)
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    initial = world(speed=0., vertical_speed=2., flight=FlightState("takeoff"))
    out = simulation(config, initial).step(VehicleControl(0., 0., 0.))
    assert not out.world.vehicle.grounded
    assert out.events[0].kind == "takeoff"
    assert out.world.vehicle.position.y == pytest.approx(.5877375)


@pytest.mark.parametrize("vector", [False, True])
def test_direct_ppo_create_cannot_bypass_research_admission(monkeypatch, vector):
    import src.rl.algorithms.ppo as ppo
    from stable_baselines3.common.vec_env import DummyVecEnv
    candidate = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION))
    env = DummyVecEnv([candidate.create]) if vector else candidate.create()
    def forbidden(*args, **kwargs):
        raise AssertionError("PPO backend created before admission")
    monkeypatch.setattr(ppo, "PPO", forbidden)
    try:
        with pytest.raises(ValueError, match="research physics training is blocked"):
            ppo.PPOModel.create(env, ppo.PPOConfig(n_steps=2, batch_size=2))
    finally:
        env.close()


def test_attached_research_environment_cannot_bypass_ppo_learn_admission():
    from src.rl.algorithms.ppo import PPOModel
    from stable_baselines3.common.vec_env import DummyVecEnv
    candidate = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION))
    env = DummyVecEnv([candidate.create])
    class Backend:
        def get_env(self):
            return env
        def learn(self, **kwargs):
            raise AssertionError("PPO learned before admission")
    try:
        with pytest.raises(ValueError, match="research physics training is blocked"):
            PPOModel(Backend()).learn(2)
    finally:
        env.close()


def test_sampled_vecenv_exposes_admission_before_first_reset():
    import src.rl.algorithms.ppo as ppo
    from src.training.config import VectorEnvConfig
    from src.training.env_factory import TrainingScenario, TrainingScenarioSet
    from src.training.vector_env import make_multi_track_vector_env
    candidate = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION))
    scenarios = TrainingScenarioSet((TrainingScenario("synthetic", candidate),))
    env = make_multi_track_vector_env(scenarios, VectorEnvConfig(n_envs=1), 9)
    try:
        with pytest.raises(ValueError, match="research physics training is blocked"):
            ppo.PPOModel.create(env, ppo.PPOConfig(n_steps=2, batch_size=2))
    finally:
        env.close()
