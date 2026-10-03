from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real


def _require_track_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("track_id must be a non-empty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_real(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _reward_components(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, tuple):
        raise ValueError("reward_component_totals must be a tuple")
    result: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("reward_component_totals entries must be (name, value) tuples")
        name, raw = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError("reward component names must be non-empty strings")
        if name in seen:
            raise ValueError("reward component names must be unique")
        seen.add(name)
        result.append((name, _require_real(raw, f"reward component {name!r}")))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    track_id: str
    success: bool
    truncated: bool
    steps: int
    episode_time: float
    completion_time: float | None
    episode_return: float
    reward_component_totals: tuple[tuple[str, float], ...]
    collision_count: int
    path_length: float
    final_normalized_route_progress: float
    mean_speed: float
    max_speed: float
    airborne_steps: int
    boost_applied_steps: int
    slow_zone_applied_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _require_track_id(self.track_id))
        success = _require_bool(self.success, "success")
        truncated = _require_bool(self.truncated, "truncated")
        _require_int(self.steps, "steps", minimum=1)
        object.__setattr__(
            self,
            "episode_time",
            _require_real(self.episode_time, "episode_time", minimum=0.0),
        )
        if self.completion_time is not None:
            object.__setattr__(
                self,
                "completion_time",
                _require_real(self.completion_time, "completion_time", minimum=0.0),
            )
        object.__setattr__(
            self,
            "episode_return",
            _require_real(self.episode_return, "episode_return"),
        )
        object.__setattr__(
            self,
            "reward_component_totals",
            _reward_components(self.reward_component_totals),
        )
        _require_int(self.collision_count, "collision_count")
        object.__setattr__(
            self,
            "path_length",
            _require_real(self.path_length, "path_length", minimum=0.0),
        )
        object.__setattr__(
            self,
            "final_normalized_route_progress",
            _require_real(
                self.final_normalized_route_progress,
                "final_normalized_route_progress",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "mean_speed",
            _require_real(self.mean_speed, "mean_speed", minimum=0.0),
        )
        object.__setattr__(
            self,
            "max_speed",
            _require_real(self.max_speed, "max_speed", minimum=0.0),
        )
        _require_int(self.airborne_steps, "airborne_steps")
        _require_int(self.boost_applied_steps, "boost_applied_steps")
        _require_int(self.slow_zone_applied_steps, "slow_zone_applied_steps")

        if success:
            if truncated:
                raise ValueError("successful episode must not be truncated")
            if self.completion_time is None:
                raise ValueError("successful episode requires completion_time")
        elif self.completion_time is not None:
            raise ValueError("unsuccessful episode must not have completion_time")


@dataclass(frozen=True, slots=True)
class TrackEvaluationSummary:
    track_id: str
    episode_count: int
    success_count: int
    success_rate: float
    mean_final_normalized_route_progress: float
    mean_episode_return: float
    mean_collision_count: float
    mean_path_length: float
    mean_speed: float
    mean_airborne_steps: float
    mean_boost_applied_steps: float
    mean_slow_zone_applied_steps: float
    mean_completion_time_for_successes: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _require_track_id(self.track_id))
        episode_count = _require_int(self.episode_count, "episode_count", minimum=1)
        success_count = _require_int(self.success_count, "success_count")
        if success_count > episode_count:
            raise ValueError("success_count must be <= episode_count")
        success_rate = _require_real(
            self.success_rate,
            "success_rate",
            minimum=0.0,
            maximum=1.0,
        )
        expected_rate = success_count / episode_count
        if not isclose(success_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("success_rate must equal success_count / episode_count")
        object.__setattr__(self, "success_rate", success_rate)

        for name, minimum, maximum in (
            ("mean_final_normalized_route_progress", 0.0, 1.0),
            ("mean_collision_count", 0.0, None),
            ("mean_path_length", 0.0, None),
            ("mean_speed", 0.0, None),
            ("mean_airborne_steps", 0.0, None),
            ("mean_boost_applied_steps", 0.0, None),
            ("mean_slow_zone_applied_steps", 0.0, None),
        ):
            object.__setattr__(
                self,
                name,
                _require_real(getattr(self, name), name, minimum=minimum, maximum=maximum),
            )
        object.__setattr__(
            self,
            "mean_episode_return",
            _require_real(self.mean_episode_return, "mean_episode_return"),
        )

        if success_count == 0:
            if self.mean_completion_time_for_successes is not None:
                raise ValueError(
                    "mean_completion_time_for_successes must be None when success_count is zero"
                )
        else:
            if self.mean_completion_time_for_successes is None:
                raise ValueError(
                    "mean_completion_time_for_successes is required when successes exist"
                )
            object.__setattr__(
                self,
                "mean_completion_time_for_successes",
                _require_real(
                    self.mean_completion_time_for_successes,
                    "mean_completion_time_for_successes",
                    minimum=0.0,
                ),
            )
