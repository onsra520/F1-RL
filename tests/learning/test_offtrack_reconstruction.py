"""C2.3b: bounded off-track corridor without penalizing ramp flight altitude."""
from dataclasses import replace

import numpy as np
import pytest

from src.learning.offtrack import OffTrackConfig, OffTrackState
from src.learning.rewards import OffTrackPenalty, RewardComposer, RewardTransition
from src.learning.route_navigation import AuthoredRoute
from src.sandbox.geometry.vector import Vec3
from tests.integration.test_authored_route_env_reconstruction import _factory


def test_corridor_uses_xz_distance_not_airborne_height() -> None:
    asset = _factory("ramp-v1", guided=True)
    route = AuthoredRoute(asset.navigation)
    ramp = route.points[3]
    assert route.horizontal_distance(ramp + Vec3(0, 40, 0)) == pytest.approx(0.0)
    assert route.horizontal_distance(ramp + Vec3(15, 40, 0)) == pytest.approx(15.0)


def test_offtrack_counter_requires_consecutive_outside_steps() -> None:
    cfg = OffTrackConfig(6.0, 3)
    state = OffTrackState()
    state = state.advance(7.0, cfg)
    assert state == OffTrackState(1, True, False)
    state = state.advance(9.0, cfg)
    assert state == OffTrackState(2, True, False)
    state = state.advance(5.9, cfg)
    assert state == OffTrackState(0, False, False)
    state = state.advance(8.0, cfg).advance(8.0, cfg).advance(8.0, cfg)
    assert state == OffTrackState(3, True, True)


@pytest.mark.parametrize("distance, steps", [(-1.0, 3), (0.0, 3), (float("nan"), 3), (6.0, 0), (6.0, -1), (6.0, True)])
def test_bad_offtrack_config_rejected(distance, steps) -> None:
    with pytest.raises(ValueError):
        OffTrackConfig(distance, steps)


def test_offtrack_penalty_requires_boolean_evidence_and_is_finite() -> None:
    factory = _factory("straight-sprint-v1", guided=True)
    state = factory.initial_state.vehicle
    term = OffTrackPenalty(2.0)
    base = RewardTransition(state,state,None,(),0.1)
    with pytest.raises(ValueError, match="off_track"):
        term.evaluate(base)
    assert term.evaluate(replace(base, off_track=True)).value == -2.0
    assert term.evaluate(replace(base, off_track=False)).value == 0.0
    for bad in (3, 0.0, "yes"):
        with pytest.raises(ValueError):
            replace(base, off_track=bad)


def test_guided_offtrack_terminates_as_unsuccessful_truncation_and_resets(monkeypatch) -> None:
    factory = replace(
        _factory("straight-sprint-v1",guided=True),
        offtrack_config=OffTrackConfig(6.0,3),
        reward_composer=RewardComposer((OffTrackPenalty(2.0),)),
    )
    env=factory.create()
    distances=iter([9.0,9.0,9.0])
    monkeypatch.setattr(AuthoredRoute, "horizontal_distance", lambda self, position: next(distances))
    env.reset()
    for i in range(1,4):
        _, reward, terminated, truncated, info = env.step(np.array([0.0,0.0],dtype=np.float32))
        assert reward == pytest.approx(-2.0)
        assert terminated is False
        assert truncated is (i==3)
        assert info["offtrack_consecutive_steps"] == i
        assert info["offtrack_lateral_meters"] == 9.0
        assert info["truncation_reason"] == ("offtrack" if i==3 else None)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0.0,0.0],dtype=np.float32))
    env.reset()
    assert env._offtrack_state == OffTrackState()


def test_offtrack_grace_resets_when_vehicle_returns_to_corridor(monkeypatch) -> None:
    factory=replace(_factory("straight-sprint-v1",guided=True),offtrack_config=OffTrackConfig(6.0,2),
        reward_composer=RewardComposer((OffTrackPenalty(3.0),)))
    env=factory.create()
    distances=iter([7.0,2.0,7.0])
    monkeypatch.setattr(AuthoredRoute,"horizontal_distance",lambda self,position:next(distances))
    env.reset()
    expected=(1,0,1)
    for streak in expected:
        _, reward, terminated, truncated, info = env.step(np.array([0,0],dtype=np.float32))
        assert info["offtrack_consecutive_steps"]==streak
        assert reward == (-3.0 if streak else 0.0)
        assert terminated is False and truncated is False


def test_offtrack_penalty_requires_guidance_and_config() -> None:
    factory=_factory("straight-sprint-v1",guided=True)
    with pytest.raises(ValueError, match="offtrack"):
        replace(factory,reward_composer=RewardComposer((OffTrackPenalty(1.0),))).create()
    with pytest.raises(ValueError, match="navigation"):
        replace(_factory("straight-sprint-v1",guided=False),offtrack_config=OffTrackConfig(6.0,2)).create()


def test_multitrack_scenarios_share_offtrack_rules_but_keep_distinct_navigation() -> None:
    from src.learning.training.env_factory import TrainingScenario, TrainingScenarioSet

    first=replace(_factory("straight-sprint-v1",guided=True),offtrack_config=OffTrackConfig(6.0,3))
    second=replace(_factory("ramp-v1",guided=True),offtrack_config=OffTrackConfig(6.0,3))
    scenarios=TrainingScenarioSet((
        TrainingScenario("straight-sprint-v1",first),
        TrainingScenario("ramp-v1",second),
    ))
    assert len(scenarios.scenarios)==2
    assert scenarios.action_space.shape==(2,)
