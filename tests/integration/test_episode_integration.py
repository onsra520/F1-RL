from pathlib import Path

import f1rl.learning as learning
from f1rl.learning.episode import EpisodeConfig, RaceProgressState, RaceRoute, advance_race_progress, episode_status
from f1rl.sandbox.entities.checkpoint import Checkpoint
from f1rl.sandbox.entities.finish_line import FinishLine
from f1rl.sandbox.geometry.rotation import Quaternion
from f1rl.sandbox.geometry.vector import Vec3
from f1rl.sandbox.interactions.state import InteractionState
from f1rl.sandbox.rules.scoring import ScoreState
from f1rl.sandbox.track.model import TrackWorld
from f1rl.sandbox.track.state import TrackState
from f1rl.sandbox.track.volume import OrientedBoxVolume
from f1rl.sandbox.vehicle import VehicleControl, VehicleParameters, VehicleState
from f1rl.sandbox.world import WorldState, step_world


def _volume(center: Vec3) -> OrientedBoxVolume:
    return OrientedBoxVolume(center, Vec3(1.0, 1.0, 1.0), Quaternion.identity())


def _events(track: TrackWorld, position: Vec3) -> tuple:
    state = WorldState(
        VehicleState(position, Vec3(0.0, 0.0, 0.0), Quaternion.identity(), Vec3(0.0, 0.0, 0.0), False),
        TrackState(frozenset()), InteractionState(frozenset(), frozenset()), ScoreState(),
    )
    parameters = VehicleParameters(2.5, 0.3, 0.6, 0.0, 20.0, 0.0, 0.0, 50.0, 9.81)
    return step_world(track, state, VehicleControl(0.0, 0.0, 0.0), parameters, 0.01, 0.5).events


def test_real_world_events_complete_overlapping_final_checkpoint_and_finish() -> None:
    center = Vec3(4.0, 0.0, 0.0)
    track = TrackWorld(checkpoints=(Checkpoint("z-final", _volume(center), 0),), finish_lines=(FinishLine("finish", _volume(center)),))
    route = RaceRoute.from_track(track)

    result = advance_race_progress(route, RaceProgressState(0, False), _events(track, center))

    assert result.state == RaceProgressState(1, True)
    assert result.completed_checkpoint_id == "z-final"
    assert result.target_position is None


def test_route_construction_uses_numeric_order_and_volume_centers() -> None:
    first, second, finish = Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0)
    track = TrackWorld(checkpoints=(Checkpoint("a-second", _volume(second), 1), Checkpoint("z-first", _volume(first), 0)), finish_lines=(FinishLine("finish", _volume(finish)),))

    route = RaceRoute.from_track(track)

    assert [(checkpoint.id, checkpoint.target_position) for checkpoint in route.checkpoints] == [("z-first", first), ("a-second", second)]
    assert route.finish.target_position == finish


def test_zero_checkpoint_real_finish_and_status_precedence_are_deterministic() -> None:
    center = Vec3(4.0, 0.0, 0.0)
    track = TrackWorld(finish_lines=(FinishLine("finish", _volume(center)),))
    route = RaceRoute.from_track(track)
    state = RaceProgressState(0, False)

    first = advance_race_progress(route, state, _events(track, center))
    second = advance_race_progress(route, state, _events(track, center))

    assert first == second
    assert first.state.finished is True
    assert episode_status(first.state, 10, EpisodeConfig(10)).terminated is True
    assert episode_status(state, 10, EpisodeConfig(10)).truncated is True
    assert track.finish_lines[0].volume.center == center


def test_public_episode_exports_preserve_existing_learning_exports() -> None:
    for name in ("RaceCheckpoint", "RaceFinish", "RaceRoute", "RaceProgressState", "RaceProgressResult", "advance_race_progress", "EpisodeConfig", "EpisodeStatus", "episode_status"):
        assert getattr(learning, name) is getattr(__import__("f1rl.learning.episode", fromlist=[name]), name)
    for name in ("ObservationConfig", "PolicyAction", "RewardTransition"):
        assert hasattr(learning, name)


def test_episode_dependency_boundary_and_sandbox_reverse_dependency() -> None:
    root = Path(__file__).parents[2]
    episode_source = "\n".join(path.read_text() for path in (root / "f1rl" / "learning" / "episode").glob("*.py"))
    for forbidden in ("f1rl.runtime", "f1rl.api", "f1rl.drivers", "f1rl.learning.observation", "f1rl.learning.action", "f1rl.learning.rewards", "gymnasium", "numpy", "torch", "fastapi", "websocket"):
        assert forbidden not in episode_source
    assert all("f1rl.learning" not in path.read_text() for path in (root / "f1rl" / "sandbox").rglob("*.py"))
