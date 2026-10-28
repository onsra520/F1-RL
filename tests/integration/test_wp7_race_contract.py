"""Grounded WP7 race semantics on synthetic tracks, never TRAIN evaluation."""
from dataclasses import replace

import numpy as np
import pytest

from src.learning.wp7_race import WP7RaceConfig
from src.learning.wp7_race import RaceContractError
from src.runtime.physics import PhysicsConfig
from src.runtime.simulation import SimulationConfig
from src.sandbox.entities.checkpoint import Checkpoint
from src.sandbox.entities.finish_line import FinishLine
from src.sandbox.track.volume import OrientedBoxVolume
from src.track_assets.model import AuthoredNavigation
from tests.integration.test_wp7_physics_integration import (
    BODY, SUSPENSION, factory, world, Vec3, Quaternion,
)


def gate(z, *, x=0., orientation=None):
    return OrientedBoxVolume(Vec3(x, .5, z), Vec3(5., 2., .001),
                             orientation or Quaternion.identity())


def race_factory(*, checkpoints=(.1, .2), finish=.3, speed=5., max_steps=10):
    base = factory(PhysicsConfig("wp7_suspension_v1", BODY, SUSPENSION), world(speed=speed))
    track = replace(base.track,
        checkpoints=tuple(Checkpoint(f"cp-{i}", gate(z), i) for i, z in enumerate(checkpoints)),
        finish_lines=(FinishLine("finish", gate(finish)),))
    points = (Vec3(0., .5, 0.), *(Vec3(0., .5, z) for z in checkpoints), Vec3(0., .5, finish))
    navigation = AuthoredNavigation(points, ("road",)*(len(points)-1), tuple(range(1, len(points)-1)))
    return replace(base, track=track, navigation=navigation,
                   episode_config=replace(base.episode_config, max_steps=max_steps),
                   simulation_config=SimulationConfig(.1, 1.), race_config=WP7RaceConfig())


def test_one_tick_crosses_thin_checkpoints_in_time_order_then_finishes_before_timeout():
    env = race_factory(max_steps=1).create()
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(np.array([0., 0.], dtype=np.float32))
        assert terminated and not truncated
        assert info["next_checkpoint_index"] == 2
        assert info["race"]["completed_checkpoint_ids"] == ("cp-0", "cp-1")
        assert info["race"]["directed_crossings"] == ("cp-0", "cp-1", "finish")
        assert info["race"]["configuration"]["version"] == "wp7_directed_grounded_v1"
    finally:
        env.close()


def test_future_checkpoint_crossed_too_early_is_not_replayed_after_expected_checkpoint():
    env = race_factory(checkpoints=(.2, .1)).create()
    try:
        env.reset()
        _, _, terminated, _, info = env.step(np.array([0., 0.], dtype=np.float32))
        assert not terminated
        assert info["next_checkpoint_index"] == 1
        assert info["race"]["completed_checkpoint_ids"] == ("cp-0",)
        assert info["race"]["directed_crossings"] == ("cp-1", "cp-0", "finish")
    finally:
        env.close()


def test_backward_gate_crossings_do_not_finish():
    from math import pi
    candidate = race_factory(checkpoints=(-.1,), finish=-.3, speed=-5., max_steps=1)
    initial = candidate.initial_state
    candidate = replace(candidate, initial_state=replace(initial, vehicle=replace(
        initial.vehicle, orientation=Quaternion.from_axis_angle(Vec3(0., 1., 0.), pi),
    )))
    env = candidate.create()
    try:
        env.reset()
        _, _, terminated, truncated, info = env.step(np.array([0., 0.], dtype=np.float32))
        assert not terminated and truncated
        assert info["next_checkpoint_index"] == 0
        assert info["race"]["directed_crossings"] == ()
    finally:
        env.close()


def test_body_overlap_alone_is_not_a_center_finish_crossing():
    env = replace(race_factory(checkpoints=(), finish=.4),
                  simulation_config=SimulationConfig(.05, 1.)).create()
    try:
        env.reset()
        assert not env.step(np.array([0., 0.], dtype=np.float32))[2]
        assert env.step(np.array([0., 0.], dtype=np.float32))[2]
    finally:
        env.close()


def sideways_factory(dt):
    from math import pi
    candidate = race_factory(checkpoints=(), finish=50.)
    state = candidate.initial_state
    return replace(candidate,
        track=replace(candidate.track, surfaces=(replace(candidate.track.surfaces[0], width=30.),)),
        initial_state=replace(state, vehicle=replace(state.vehicle,
            velocity=Vec3(5., 0., 0.),
            orientation=Quaternion.from_axis_angle(Vec3(0., 1., 0.), pi/2),)),
        simulation_config=SimulationConfig(dt, 1.))


def test_soft_corridor_is_telemetry_without_invented_reward_and_hard_violation_invalidates():
    env = sideways_factory(.5).create()
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(np.array([0., 0.], dtype=np.float32))
        assert info["race"]["soft_zone_encountered"]
        assert info["race"]["max_sampled_deviation_m"] == pytest.approx(2.5)
        assert reward == 0.
        assert not terminated and not truncated
        with pytest.raises(RaceContractError, match="hard_violation"):
            env.step(np.array([0., 0.], dtype=np.float32))
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.array([0., 0.], dtype=np.float32))
        assert env.reset()[1]["next_checkpoint_index"] == 0
    finally:
        env.close()


def test_exact_four_meter_boundary_is_allowed():
    env = sideways_factory(.8).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        assert info["race"]["max_sampled_deviation_m"] == pytest.approx(4.)
    finally:
        env.close()


def test_simultaneous_finish_and_checkpoint_order_is_rejected():
    candidate = race_factory(checkpoints=(.1,), finish=.3)
    finish = replace(candidate.track.finish_lines[0], volume=gate(.1))
    # A non-adjacent return segment keeps navigation valid while making the
    # finish and checkpoint planes simultaneous: no invented tie-break.
    navigation = AuthoredNavigation(
        (Vec3(0., .5, 0.), Vec3(0., .5, .1), Vec3(0., .5, .2), Vec3(0., .5, .1)),
        ("road", "road", "road"), (1,),
    )
    env = replace(candidate, track=replace(candidate.track, finish_lines=(finish,)),
                  navigation=navigation).create()
    try:
        env.reset()
        with pytest.raises(RaceContractError, match="simultaneous gate order"):
            env.step(np.array([0., 0.], dtype=np.float32))
    finally:
        env.close()


def test_race_contract_cannot_mix_legacy_or_airborne_or_c2_semantics():
    from src.learning.offtrack import OffTrackConfig
    from src.sandbox.ballistic_flight import BallisticFlightPolicy, FlightState
    candidate = race_factory()
    with pytest.raises(ValueError, match="grounded physics"):
        replace(candidate, physics_config=None)
    with pytest.raises(ValueError, match="without C2"):
        replace(candidate, offtrack_config=OffTrackConfig(6., 10))
    flight = BallisticFlightPolicy(SUSPENSION.contact, 3., .1)
    config = PhysicsConfig("wp7_ballistic_v1", BODY, SUSPENSION, flight)
    with pytest.raises(ValueError, match="grounded physics"):
        replace(candidate, physics_config=config,
                initial_state=replace(candidate.initial_state, flight=FlightState("grounded")))
    with pytest.raises(ValueError, match="changed owner limits"):
        WP7RaceConfig(hard_corridor_m=6.)


def test_invalid_body_collision_takes_precedence_over_finish_and_requires_reset():
    from src.sandbox.entities.obstacle import Obstacle
    candidate = race_factory(max_steps=1)
    wall = Obstacle("wall", OrientedBoxVolume(
        Vec3(0., .8, 2.2), Vec3(.5, .8, .02), Quaternion.identity(),
    ))
    env = replace(candidate, track=replace(candidate.track, obstacles=(wall,))).create()
    try:
        env.reset()
        with pytest.raises(RuntimeError, match="collision recovery is unregistered"):
            env.step(np.array([0., 0.], dtype=np.float32))
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.array([0., 0.], dtype=np.float32))
    finally:
        env.close()
