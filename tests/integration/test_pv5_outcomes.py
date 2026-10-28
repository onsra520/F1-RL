import numpy as np
import pytest

from tests.integration.test_wp7_race_contract import race_factory, sideways_factory
from src.learning.wp7_race import RaceContractError
from src.runtime.physics import PhysicsTransitionError


@pytest.mark.parametrize("finish,max_steps,kind", [(.3, 1, "successful_finish"),
                                                   (50., 1, "valid_truncation")])
def test_accepted_outcome_is_machine_readable_without_changing_terminal_flags(finish, max_steps, kind):
    env = race_factory(finish=finish, max_steps=max_steps).create()
    try:
        env.reset()
        info = env.step(np.array([0., 0.], dtype=np.float32))[-1]
        assert info["research_outcome"]["kind"] == kind
        assert info["research_outcome"]["valid_sample"]
    finally:
        env.close()


def test_uncertified_route_never_returns_a_ppo_sample():
    env = sideways_factory(1.).create()
    try:
        env.reset()
        with pytest.raises(RaceContractError):
            env.step(np.array([0., 0.], dtype=np.float32))
        assert env.last_research_outcome["kind"] == "uncertified_route_state"
        assert not env.last_research_outcome["valid_sample"]
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.array([0., 0.], dtype=np.float32))
    finally:
        env.close()


def test_invalid_physics_has_precedence_and_is_not_terminal_failure(monkeypatch):
    env = race_factory(max_steps=1).create()
    from src.runtime.simulation import Simulation
    def reject(*args, **kwargs):
        raise PhysicsTransitionError("unregistered recovery")
    try:
        env.reset()
        monkeypatch.setattr(Simulation, "step", reject)
        with pytest.raises(PhysicsTransitionError):
            env.step(np.array([0., 0.], dtype=np.float32))
        assert env.last_research_outcome["kind"] == "invalid_physical_transition"
        assert not env.last_research_outcome["valid_sample"]
    finally:
        env.close()
