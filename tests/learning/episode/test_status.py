from dataclasses import FrozenInstanceError

import pytest

from f1rl.learning.episode import EpisodeConfig, EpisodeStatus, RaceProgressState, episode_status


@pytest.mark.parametrize("max_steps", [True, 0, -1, 1.0, "1", None])
def test_episode_config_rejects_non_positive_integers(max_steps: object) -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(max_steps)


def test_episode_config_is_immutable() -> None:
    config = EpisodeConfig(10)

    with pytest.raises(FrozenInstanceError):
        config.max_steps = 20


@pytest.mark.parametrize("progress, step_count, config", [("progress", 0, EpisodeConfig(1)), (RaceProgressState(0, False), 0, "config"), (RaceProgressState(0, False), True, EpisodeConfig(1)), (RaceProgressState(0, False), -1, EpisodeConfig(1)), (RaceProgressState(0, False), 1.0, EpisodeConfig(1))])
def test_episode_status_rejects_invalid_inputs(progress: object, step_count: object, config: object) -> None:
    with pytest.raises(ValueError):
        episode_status(progress, step_count, config)


@pytest.mark.parametrize(
    "progress, step_count, expected",
    [
        (RaceProgressState(0, False), 9, EpisodeStatus(False, False)),
        (RaceProgressState(0, False), 10, EpisodeStatus(False, True)),
        (RaceProgressState(0, False), 11, EpisodeStatus(False, True)),
        (RaceProgressState(0, True), 10, EpisodeStatus(True, False)),
    ],
)
def test_episode_status_applies_finish_before_timeout(progress: RaceProgressState, step_count: int, expected: EpisodeStatus) -> None:
    assert episode_status(progress, step_count, EpisodeConfig(10)) == expected


def test_episode_status_is_deterministic_and_does_not_mutate_inputs() -> None:
    progress = RaceProgressState(0, False)
    config = EpisodeConfig(10)

    assert episode_status(progress, 10, config) == episode_status(progress, 10, config) == EpisodeStatus(False, True)
    assert progress == RaceProgressState(0, False)
    assert config == EpisodeConfig(10)
