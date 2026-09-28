from dataclasses import dataclass

from .progress import RaceProgressState


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    max_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")


@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    terminated: bool
    truncated: bool


def episode_status(progress: RaceProgressState, step_count: int, config: EpisodeConfig) -> EpisodeStatus:
    if not isinstance(progress, RaceProgressState):
        raise ValueError("progress must be RaceProgressState")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
        raise ValueError("step_count must be a non-negative integer")
    if not isinstance(config, EpisodeConfig):
        raise ValueError("config must be EpisodeConfig")
    if progress.finished:
        return EpisodeStatus(True, False)
    return EpisodeStatus(False, step_count >= config.max_steps)
