from dataclasses import dataclass
from math import inf

import pytest

from src.rl.rewards.core import RewardComposer, RewardContribution, RewardResult, RewardTransition
from src.simulation.geometry.rotation import Quaternion
from src.simulation.geometry.vector import Vec3
from src.simulation.vehicle import VehicleState


def vehicle(z: float = 0.0) -> VehicleState:
    return VehicleState(Vec3(0.0, 0.3, z), Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), True)


def transition() -> RewardTransition:
    return RewardTransition(vehicle(), vehicle(1.0), Vec3(0.0, 0.3, 10.0), (), 0.1)


@dataclass
class FakeTerm:
    name: str
    value: float
    calls: list[str]

    def evaluate(self, transition: RewardTransition) -> RewardContribution:
        self.calls.append(self.name)
        return RewardContribution(self.name, self.value)


def test_composer_validates_term_container_and_contract() -> None:
    with pytest.raises(ValueError):
        RewardComposer([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RewardComposer((object(),))
    with pytest.raises(ValueError):
        RewardComposer((FakeTerm("", 1.0, []),))
    with pytest.raises(ValueError):
        RewardComposer((FakeTerm("dup", 1.0, []), FakeTerm("dup", 2.0, [])))


def test_composer_preserves_order_calls_once_and_does_not_clip() -> None:
    calls: list[str] = []
    terms = (FakeTerm("zeta", 3.0, calls), FakeTerm("alpha", -5.5, calls), FakeTerm("middle", 1.25, calls))
    result = RewardComposer(terms).evaluate(transition())
    assert calls == ["zeta", "alpha", "middle"]
    assert result.contributions == (
        RewardContribution("zeta", 3.0),
        RewardContribution("alpha", -5.5),
        RewardContribution("middle", 1.25),
    )
    assert result.total == pytest.approx(-1.25)


def test_composer_rejects_invalid_term_outputs() -> None:
    class WrongType:
        name = "wrong-type"
        def evaluate(self, transition):
            return 1.0

    class WrongName:
        name = "declared"
        def evaluate(self, transition):
            return RewardContribution("other", 1.0)

    class NonFinite:
        name = "non-finite"
        def evaluate(self, transition):
            broken = object.__new__(RewardContribution)
            object.__setattr__(broken, "name", self.name)
            object.__setattr__(broken, "value", inf)
            return broken

    for term in (WrongType(), WrongName(), NonFinite()):
        with pytest.raises(ValueError):
            RewardComposer((term,)).evaluate(transition())


def test_empty_composer_and_determinism() -> None:
    subject = transition()
    assert RewardComposer(()).evaluate(subject) == RewardResult(0.0, ())
    composer = RewardComposer((FakeTerm("a", 1.0, []), FakeTerm("b", -0.25, [])))
    first = composer.evaluate(subject)
    second = composer.evaluate(subject)
    assert first == second == RewardResult(0.75, (RewardContribution("a", 1.0), RewardContribution("b", -0.25)))
    assert subject == transition()
