"""C2.3 reconstructed high-water route reward, without old target-distance shaping."""
from dataclasses import replace
from math import nan

import numpy as np
import pytest

from src.learning.rewards import RewardComposer, RouteProgressReward, RewardTransition
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.vehicle import VehicleState


def _state(z=0.0):
    return VehicleState(
        Vec3(0,0,z),Vec3(0,0,0),Quaternion.identity(),Vec3(0,0,0),True
    )


def _transition(delta=None):
    return RewardTransition(
        _state(),_state(2.0),None,(),0.1,
        route_progress_delta_meters=delta,
    )


def test_route_reward_uses_validated_highwater_delta_only() -> None:
    term=RouteProgressReward(2.0)
    assert term.name == "route_progress"
    assert term.evaluate(_transition(3.5)).value == pytest.approx(7.0)
    assert term.evaluate(_transition(0.0)).value == 0.0
    assert RewardComposer((term,)).evaluate(_transition(1.2)).total == pytest.approx(2.4)


def test_route_reward_rejects_missing_highwater_evidence() -> None:
    with pytest.raises(ValueError,match="route"):
        RouteProgressReward(1.0).evaluate(_transition(None))


@pytest.mark.parametrize("delta", [nan,-1.0,True,"0.1"])
def test_route_transition_rejects_invalid_delta(delta) -> None:
    with pytest.raises(ValueError,match="route"):
        _transition(delta)


@pytest.mark.parametrize("weight", [nan,-1.0,True,"1"])
def test_route_reward_rejects_invalid_weight(weight) -> None:
    with pytest.raises(ValueError):
        RouteProgressReward(weight)


def test_legacy_transitions_still_support_ordinary_reward_terms() -> None:
    from src.learning.rewards import TimePenalty
    assert TimePenalty(0.1).evaluate(_transition()).value == pytest.approx(-0.01)


def test_guided_environment_tracks_route_highwater_across_steps_and_resets() -> None:
    from tests.integration.test_authored_route_env_reconstruction import _factory

    factory=replace(
        _factory("straight-sprint-v1",guided=True),
        reward_composer=RewardComposer((RouteProgressReward(1.0),)),
    )
    env=factory.create()
    env.reset()
    assert env._route_highwater == pytest.approx(0.0)
    _obs,first,terminated,truncated,info=env.step(np.array([0.0,1.0],dtype=np.float32))
    assert first >= 0.0
    assert info["route_highwater_meters"] >= 0.0
    assert info["reward_components"]["route_progress"] == pytest.approx(first)
    previous=env._route_highwater
    _obs,second,terminated,truncated,info=env.step(np.array([0.0,1.0],dtype=np.float32))
    assert env._route_highwater >= previous
    assert second == pytest.approx(env._route_highwater-previous)
    assert info["route_highwater_meters"] == pytest.approx(env._route_highwater)
    assert not terminated and not truncated
    env.reset()
    assert env._route_highwater == pytest.approx(0.0)


def test_route_reward_without_authored_navigation_is_configuration_error() -> None:
    from tests.integration.test_authored_route_env_reconstruction import _factory

    factory=replace(
        _factory("straight-sprint-v1",guided=False),
        reward_composer=RewardComposer((RouteProgressReward(1.0),)),
    )
    with pytest.raises(ValueError,match="navigation"):
        factory.create()
