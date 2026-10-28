from dataclasses import FrozenInstanceError

import pytest

from src.evaluation import EpisodeMetrics, TrackEvaluationSummary


def _episode(**overrides: object) -> EpisodeMetrics:
    values: dict[str, object] = {
        "track_id": "track-a",
        "success": True,
        "truncated": False,
        "steps": 12,
        "episode_time": 1.2,
        "completion_time": 1.2,
        "episode_return": 3.5,
        "reward_component_totals": (
            ("target_progress", 4.7),
            ("collision", 0.0),
            ("time", -1.2),
        ),
        "collision_count": 0,
        "path_length": 10.0,
        "final_normalized_route_progress": 1.0,
        "mean_speed": 8.0,
        "max_speed": 12.0,
        "airborne_steps": 0,
        "boost_applied_steps": 2,
        "slow_zone_applied_steps": 0,
    }
    values.update(overrides)
    return EpisodeMetrics(**values)  # type: ignore[arg-type]


def _summary(**overrides: object) -> TrackEvaluationSummary:
    values: dict[str, object] = {
        "track_id": "track-a",
        "episode_count": 2,
        "success_count": 1,
        "success_rate": 0.5,
        "mean_final_normalized_route_progress": 0.75,
        "mean_episode_return": -1.25,
        "mean_collision_count": 1.0,
        "mean_path_length": 12.5,
        "mean_speed": 7.0,
        "mean_airborne_steps": 0.5,
        "mean_boost_applied_steps": 2.0,
        "mean_slow_zone_applied_steps": 1.0,
        "mean_completion_time_for_successes": 8.0,
    }
    values.update(overrides)
    return TrackEvaluationSummary(**values)  # type: ignore[arg-type]


def test_episode_metrics_is_frozen_slotted_and_keeps_exact_contract() -> None:
    metrics = _episode()

    assert metrics.track_id == "track-a"
    assert metrics.reward_component_totals == (
        ("target_progress", 4.7),
        ("collision", 0.0),
        ("time", -1.2),
    )
    assert hasattr(EpisodeMetrics, "__slots__")
    with pytest.raises(FrozenInstanceError):
        metrics.steps = 13  # type: ignore[misc]


@pytest.mark.parametrize("track_id", ["", "   ", 123, None])
def test_episode_metrics_rejects_invalid_track_id(track_id: object) -> None:
    with pytest.raises(ValueError):
        _episode(track_id=track_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", 1),
        ("truncated", 0),
        ("steps", True),
        ("steps", 0),
        ("steps", -1),
        ("collision_count", True),
        ("collision_count", -1),
        ("airborne_steps", -1),
        ("boost_applied_steps", -1),
        ("slow_zone_applied_steps", -1),
    ],
)
def test_episode_metrics_rejects_invalid_bool_or_count_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _episode(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_time", -0.1),
        ("episode_time", float("nan")),
        ("episode_time", float("inf")),
        ("path_length", -0.1),
        ("path_length", float("nan")),
        ("mean_speed", -0.1),
        ("max_speed", -0.1),
        ("episode_return", float("nan")),
        ("episode_return", float("inf")),
    ],
)
def test_episode_metrics_rejects_invalid_scalar_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _episode(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_episode_metrics_rejects_invalid_normalized_progress(value: float) -> None:
    with pytest.raises(ValueError):
        _episode(final_normalized_route_progress=value)


def test_episode_metrics_validates_completion_semantics() -> None:
    with pytest.raises(ValueError):
        _episode(success=True, truncated=True)
    with pytest.raises(ValueError):
        _episode(success=True, completion_time=None)
    with pytest.raises(ValueError):
        _episode(success=False, truncated=True, completion_time=1.2)

    failed = _episode(success=False, truncated=True, completion_time=None)
    assert failed.completion_time is None


@pytest.mark.parametrize(
    "components",
    [
        [("target_progress", 1.0)],
        (("", 1.0),),
        (("   ", 1.0),),
        (("same", 1.0), ("same", 2.0)),
        (("target_progress", float("nan")),),
    ],
)
def test_episode_metrics_rejects_invalid_reward_components(components: object) -> None:
    with pytest.raises(ValueError):
        _episode(reward_component_totals=components)


def test_track_summary_is_frozen_slotted_and_validates_success_rate() -> None:
    summary = _summary()
    assert summary.success_rate == 0.5
    assert hasattr(TrackEvaluationSummary, "__slots__")
    with pytest.raises(FrozenInstanceError):
        summary.episode_count = 3  # type: ignore[misc]

    with pytest.raises(ValueError):
        _summary(success_rate=0.4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("track_id", ""),
        ("episode_count", 0),
        ("episode_count", True),
        ("success_count", -1),
        ("success_count", 3),
        ("success_count", True),
        ("success_rate", -0.1),
        ("success_rate", 1.1),
        ("mean_final_normalized_route_progress", -0.1),
        ("mean_final_normalized_route_progress", 1.1),
        ("mean_collision_count", -0.1),
        ("mean_path_length", -0.1),
        ("mean_speed", -0.1),
        ("mean_airborne_steps", -0.1),
        ("mean_boost_applied_steps", -0.1),
        ("mean_slow_zone_applied_steps", -0.1),
    ],
)
def test_track_summary_rejects_invalid_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _summary(**{field: value})


def test_track_summary_completion_time_matches_success_count() -> None:
    with pytest.raises(ValueError):
        _summary(success_count=0, success_rate=0.0, mean_completion_time_for_successes=8.0)
    with pytest.raises(ValueError):
        _summary(success_count=1, success_rate=0.5, mean_completion_time_for_successes=None)

    summary = _summary(
        success_count=0,
        success_rate=0.0,
        mean_completion_time_for_successes=None,
    )
    assert summary.mean_completion_time_for_successes is None
