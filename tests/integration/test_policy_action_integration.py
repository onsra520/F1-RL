from __future__ import annotations

import ast
from pathlib import Path

import src.learning as learning
import src.learning.action as action_module
import src.learning.observation as observation_module
from src.learning.action import PolicyAction, map_policy_action
from src.runtime.simulation import Simulation, SimulationConfig
from src.sandbox.geometry.rotation import Quaternion
from src.sandbox.geometry.vector import Vec3
from src.sandbox.interactions.state import InteractionState
from src.sandbox.rules.scoring import ScoreState
from src.sandbox.track.model import TrackWorld
from src.sandbox.track.patch import PlanarSurfacePatch
from src.sandbox.track.state import TrackState
from src.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from src.sandbox.world import WorldState


def _road() -> PlanarSurfacePatch:
    return PlanarSurfacePatch(
        "road", Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 1.0, 0.0), 10.0, 100.0,
    )


def _world() -> WorldState:
    vehicle = VehicleState(
        Vec3(0.0, 0.3, 0.0),
        Vec3(0.0, 0.0, 6.0),
        Quaternion.identity(),
        Vec3(0.0, 0.0, 0.0),
        True,
    )
    return WorldState(
        vehicle,
        TrackState(frozenset()),
        InteractionState(frozenset(), frozenset()),
        ScoreState(),
    )


def _parameters() -> VehicleParameters:
    return VehicleParameters(2.5, 0.3, 0.6, 8.0, 12.0, 0.1, 0.01, 50.0, 9.81)


def _simulation() -> Simulation:
    return Simulation(
        TrackWorld(surfaces=(_road(),)),
        _world(),
        _parameters(),
        SimulationConfig(0.1, 0.5),
    )


def test_mapped_action_matches_direct_control_for_one_step() -> None:
    mapped_simulation = _simulation()
    direct_simulation = _simulation()
    policy_action = PolicyAction(0.35, 0.8)
    mapped_control = map_policy_action(policy_action)
    direct_control = VehicleControl(0.35, 0.8, 0.0)

    assert mapped_control == direct_control
    mapped_snapshot = mapped_simulation.step(mapped_control)
    direct_snapshot = direct_simulation.step(direct_control)
    assert mapped_snapshot == direct_snapshot


def test_public_action_exports_match_authoritative_definitions() -> None:
    assert learning.PolicyAction is action_module.PolicyAction
    assert learning.map_policy_action is action_module.map_policy_action
    assert learning.ACTION_NAMES == action_module.ACTION_NAMES
    assert learning.ACTION_LOW == action_module.ACTION_LOW
    assert learning.ACTION_HIGH == action_module.ACTION_HIGH
    assert learning.ACTION_SIZE == action_module.ACTION_SIZE
    assert learning.NavigationTarget is observation_module.NavigationTarget
    assert learning.ObservationConfig is observation_module.ObservationConfig
    assert learning.ObservationEncoder is observation_module.ObservationEncoder


def _imported_modules(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append((node.level, node.module))
    return tuple(modules)


def test_action_module_has_no_forbidden_dependencies() -> None:
    path = Path("src/learning/action.py")
    imports = _imported_modules(path)
    forbidden = (
        "runtime", "api", "drivers", "observation", "rewards", "gymnasium",
        "numpy", "torch", "stable_baselines3", "fastapi", "websocket", "websockets",
    )
    for _, module in imports:
        assert not any(module == item or module.startswith(item + ".") for item in forbidden)

    source = path.read_text(encoding="utf-8").lower()
    assert not any(token in source for token in ("ppo", "react", "three.js"))


def test_sandbox_does_not_import_learning() -> None:
    offenders: list[str] = []
    for path in Path("src/sandbox").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "src.learning" or alias.name.startswith("src.learning.")
                    for alias in node.names
                ):
                    offenders.append(str(path))
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if module == "src.learning" or module.startswith("src.learning."):
                    offenders.append(str(path))
                if node.level > 0 and (module == "learning" or module.startswith("learning.")):
                    offenders.append(str(path))
    assert offenders == []
