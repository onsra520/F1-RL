import pytest

from f1rl.learning.evaluation import EpisodeMetrics, summarize_track


def _episode(
    *,
    success: bool,
    steps: int,
    episode_time: float,
    completion_time: float | None,
    episode_return: float,
    collisions: int,
    path_length: float,
    progress: float,
    mean_speed: float,
    max_speed: float,
    airborne_steps: int,
    boost_steps: int,
    slow_steps: int,
    track_id: str = "track-a",
) -> EpisodeMetrics:
    return EpisodeMetrics(
        track_id=track_id,
        success=success,
        truncated=not success,
        steps=steps,
        episode_time=episode_time,
        completion_time=completion_time,
        episode_return=episode_return,
        reward_component_totals=(("total", episode_return),),
        collision_count=collisions,
        path_length=path_length,
        final_normalized_route_progress=progress,
        mean_speed=mean_speed,
        max_speed=max_speed,
        airborne_steps=airborne_steps,
        boost_applied_steps=boost_steps,
        slow_zone_applied_steps=slow_steps,
    )


def test_summarize_track_uses_episode_level_descriptive_means() -> None:
    episodes = (
        _episode(
            success=True,
            steps=2,
            episode_time=8.0,
            completion_time=8.0,
            episode_return=4.0,
            collisions=0,
            path_length=10.0,
            progress=1.0,
            mean_speed=10.0,
            max_speed=12.0,
            airborne_steps=0,
            boost_steps=2,
            slow_steps=0,
        ),
        _episode(
            success=False,
            steps=100,
            episode_time=10.0,
            completion_time=None,
            episode_return=-2.0,
            collisions=2,
            path_length=20.0,
            progress=0.5,
            mean_speed=2.0,
            max_speed=5.0,
            airborne_steps=4,
            boost_steps=0,
            slow_steps=2,
        ),
    )

    summary = summarize_track(episodes)

    assert summary.track_id == "track-a"
    assert summary.episode_count == 2
    assert summary.success_count == 1
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_final_normalized_route_progress == pytest.approx(0.75)
    assert summary.mean_episode_return == pytest.approx(1.0)
    assert summary.mean_collision_count == pytest.approx(1.0)
    assert summary.mean_path_length == pytest.approx(15.0)
    assert summary.mean_speed == pytest.approx(6.0)
    assert summary.mean_airborne_steps == pytest.approx(2.0)
    assert summary.mean_boost_applied_steps == pytest.approx(1.0)
    assert summary.mean_slow_zone_applied_steps == pytest.approx(1.0)
    assert summary.mean_completion_time_for_successes == pytest.approx(8.0)


def test_summarize_track_returns_no_completion_mean_without_successes() -> None:
    episodes = (
        _episode(
            success=False,
            steps=5,
            episode_time=0.5,
            completion_time=None,
            episode_return=-1.0,
            collisions=0,
            path_length=1.0,
            progress=0.2,
            mean_speed=2.0,
            max_speed=3.0,
            airborne_steps=0,
            boost_steps=0,
            slow_steps=0,
        ),
    )
    assert summarize_track(episodes).mean_completion_time_for_successes is None


@pytest.mark.parametrize(
    "episodes",
    [
        (),
        ("not-metrics",),
    ],
)
def test_summarize_track_rejects_invalid_episode_collection(episodes: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        summarize_track(episodes)  # type: ignore[arg-type]


def test_summarize_track_rejects_mixed_track_ids() -> None:
    first = _episode(
        success=False,
        steps=5,
        episode_time=0.5,
        completion_time=None,
        episode_return=0.0,
        collisions=0,
        path_length=1.0,
        progress=0.2,
        mean_speed=2.0,
        max_speed=3.0,
        airborne_steps=0,
        boost_steps=0,
        slow_steps=0,
    )
    second = _episode(
        success=False,
        steps=5,
        episode_time=0.5,
        completion_time=None,
        episode_return=0.0,
        collisions=0,
        path_length=1.0,
        progress=0.2,
        mean_speed=2.0,
        max_speed=3.0,
        airborne_steps=0,
        boost_steps=0,
        slow_steps=0,
        track_id="track-b",
    )

    with pytest.raises(ValueError, match="track_id"):
        summarize_track((first, second))
