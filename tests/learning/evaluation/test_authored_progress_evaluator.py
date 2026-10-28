"""C2 evaluation must use authoritative high-water route progress, not old target chords."""
from dataclasses import replace

import numpy as np
import pytest

from src.learning.evaluation import evaluate_episode
from src.learning.training.env_factory import F1RLEnvFactory
from tests.integration.test_authored_route_env_reconstruction import _factory
from tests.learning.evaluation.mocks.environments import ScriptedEvaluationEnv
from tests.learning.evaluation.mocks.policies import ConstantPolicy


def _step(tick: int, highwater: float | None, *, position: float, missing: bool = False):
    info={
        "tick": tick, "elapsed_time": tick*0.1,
        "next_checkpoint_index": 0, "finished": False,
        "reward_components": {"route_progress": 0.0},
        "telemetry": {
            "vehicle_position": (0.0, 0.0, float(position)),
            "vehicle_speed": 1.0, "grounded": True,
            "obstacle_hit_ids": (), "applied_boost_pad_ids": (),
            "applied_slow_zone_ids": (),
        },
    }
    if not missing:
        info["route_highwater_meters"]=highwater
    return (np.zeros(16,dtype=np.float32), 0.0, False, tick==2, info)


def _evaluate(monkeypatch, steps):
    env=ScriptedEvaluationEnv(
        {"telemetry":{"vehicle_position":(0.0,0.0,4.0)}}, tuple(steps),
    )
    monkeypatch.setattr(F1RLEnvFactory,"create",lambda self: env)
    factory=_factory("straight-sprint-v1",guided=True)
    return evaluate_episode(
        track_id="straight-sprint-v1",factory=factory,
        policy=ConstantPolicy(np.array([0,0],dtype=np.float32))
    )


def test_guided_evaluation_reports_authoritative_highwater_not_final_chord(monkeypatch):
    result=_evaluate(monkeypatch,(
        _step(1,5.0,position=9.0),
        _step(2,5.0,position=6.0),
    ))
    assert result.final_normalized_route_progress==pytest.approx(5.0/108.0)
    assert result.truncated and not result.success


@pytest.mark.parametrize("steps",[
    (_step(1,5.0,position=9.0),_step(2,4.0,position=6.0)),
    (_step(1,None,position=9.0),_step(2,5.0,position=6.0)),
    (_step(1,109.0,position=9.0),_step(2,109.0,position=6.0)),
    (_step(1,33.0,position=9.0),_step(2,33.0,position=6.0)),
    (_step(1,5.0,position=9.0,missing=True),_step(2,5.0,position=6.0)),
])
def test_guided_evaluation_rejects_missing_decreasing_or_impossible_highwater(monkeypatch,steps):
    with pytest.raises(ValueError,match="route_highwater"):
        _evaluate(monkeypatch,steps)
