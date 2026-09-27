from __future__ import annotations

import ast
from pathlib import Path

import pytest

import f1rl.learning as learning
import f1rl.learning.rewards as rewards
from f1rl.learning.rewards import CollisionPenalty, RewardComposer, RewardContribution, RewardResult, RewardTransition, TargetProgressReward, TimePenalty
from f1rl.runtime.simulation import Simulation, SimulationConfig
from f1rl.sandbox.entities.obstacle import Obstacle
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.events import CheckpointEntered, FinishLineEntered, ObstacleHit
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.patch import PlanarSurfacePatch
from f1rl.sandbox.track.state import ScorePickupCollected, TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState


def road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch("road", Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0), 10.0, 100.0)


def world(speed: float = 5.0) -> WorldState:
    return WorldState(
        VehicleState(Vec3(0, 0.3, 0), Vec3(0, 0, speed), Quaternion.identity(), Vec3(0, 0, 0), True),
        TrackState(frozenset()), InteractionState(frozenset(), frozenset()), ScoreState(),
    )


def parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 0.0, 12.0, 0.0, 0.0, 100.0, 9.81)


def simulation(track: TrackWorld | None = None, speed: float = 5.0) -> Simulation:
    return Simulation(track or TrackWorld(surfaces=(road(),)), world(speed), parameters(), SimulationConfig(0.1, 0.5))


def standard_composer() -> RewardComposer:
    return RewardComposer((TargetProgressReward(1.0), CollisionPenalty(5.0), TimePenalty(0.1)))


def test_real_simulation_progress_uses_fixed_pre_step_target() -> None:
    toward = simulation()
    before_toward = toward.snapshot
    target_toward = Vec3(0, 0.3, 10)
    after_toward = toward.step(VehicleControl(0, 0, 0))
    toward_result = standard_composer().evaluate(RewardTransition(before_toward.world.vehicle, after_toward.world.vehicle, target_toward, after_toward.events, 0.1))
    assert toward_result.contributions[0].name == "target_progress"
    assert toward_result.contributions[0].value > 0.0
    assert toward_result.contributions[2].name == "time"
    assert toward_result.contributions[2].value == pytest.approx(-0.01)

    away = simulation()
    before_away = away.snapshot
    target_away = Vec3(0, 0.3, -10)
    after_away = away.step(VehicleControl(0, 0, 0))
    away_result = standard_composer().evaluate(RewardTransition(before_away.world.vehicle, after_away.world.vehicle, target_away, after_away.events, 0.1))
    assert away_result.contributions[0].value < 0.0


def test_overlapping_obstacles_produce_one_collision_penalty() -> None:
    volume = OrientedBoxVolume(Vec3(0, 0.3, 1.0), Vec3(1.0, 1.0, 0.2), Quaternion.identity())
    track = TrackWorld(surfaces=(road(),), obstacles=(Obstacle("b", volume), Obstacle("a", volume)))
    sim = simulation(track, speed=10.0)
    before = sim.snapshot
    after = sim.step(VehicleControl(0, 0, 0))
    assert after.events == (ObstacleHit("a"), ObstacleHit("b"))
    result = standard_composer().evaluate(RewardTransition(before.world.vehicle, after.world.vehicle, Vec3(0, 0.3, 10), after.events, 0.1))
    assert result.contributions[1] == RewardContribution("collision", -5.0)


def test_checkpoint_finish_and_pickup_events_have_no_implicit_bonus() -> None:
    vehicle = world(0.0).vehicle
    events = (CheckpointEntered("cp", 0), FinishLineEntered("finish"), ScorePickupCollected("card", 1000))
    result = standard_composer().evaluate(RewardTransition(vehicle, vehicle, Vec3(0, 0.3, 10), events, 0.1))
    assert tuple(item.name for item in result.contributions) == ("target_progress", "collision", "time")
    assert tuple(item.value for item in result.contributions) == pytest.approx((0.0, 0.0, -0.01))
    assert result.total == pytest.approx(-0.01)


def test_independent_equal_transitions_are_deterministic_and_non_mutating() -> None:
    sim_a, sim_b = simulation(), simulation()
    before_a, before_b = sim_a.snapshot, sim_b.snapshot
    after_a = sim_a.step(VehicleControl(0, 0, 0))
    after_b = sim_b.step(VehicleControl(0, 0, 0))
    target = Vec3(0, 0.3, 10)
    transition_a = RewardTransition(before_a.world.vehicle, after_a.world.vehicle, target, after_a.events, 0.1)
    transition_b = RewardTransition(before_b.world.vehicle, after_b.world.vehicle, target, after_b.events, 0.1)
    result_a = standard_composer().evaluate(transition_a)
    result_b = standard_composer().evaluate(transition_b)
    assert result_a == result_b
    assert transition_a == transition_b
    assert sim_a.snapshot == after_a and sim_b.snapshot == after_b


def test_public_reward_exports_preserve_existing_learning_exports() -> None:
    names = ("RewardTransition", "RewardContribution", "RewardResult", "RewardTerm", "RewardComposer", "TargetProgressReward", "CollisionPenalty", "TimePenalty")
    for name in names:
        assert getattr(learning, name) is getattr(rewards, name)
    assert learning.PolicyAction is not None
    assert learning.ObservationEncoder is not None


def _imports(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.level, node.module))
    return tuple(found)


def test_reward_modules_have_no_forbidden_dependencies() -> None:
    forbidden = ("runtime", "api", "drivers", "observation", "action", "gymnasium", "numpy", "torch", "stable_baselines3", "fastapi", "websocket", "websockets")
    for path in Path("f1rl/learning/rewards").glob("*.py"):
        for _, module in _imports(path):
            assert not any(module == item or module.startswith(item + ".") for item in forbidden), (path, module)
        source = path.read_text(encoding="utf-8-sig").lower()
        assert not any(token in source for token in ("ppo", "react", "three.js", "scorestate"))


def test_sandbox_does_not_import_learning() -> None:
    offenders: list[str] = []
    for path in Path("f1rl/sandbox").rglob("*.py"):
        for level, module in _imports(path):
            if module == "f1rl.learning" or module.startswith("f1rl.learning."):
                offenders.append(str(path))
            if level > 0 and (module == "learning" or module.startswith("learning.")):
                offenders.append(str(path))
    assert offenders == []

