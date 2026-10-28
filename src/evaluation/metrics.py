from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, sqrt
from numbers import Real
from statistics import fmean

from src.rl.episode import RaceRoute


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


def _position_tuple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-tuple")
    return (
        _require_real(value[0], f"{name}[0]"),
        _require_real(value[1], f"{name}[1]"),
        _require_real(value[2], f"{name}[2]"),
    )


def _route_point(value: object) -> tuple[float, float, float]:
    return (
        _require_real(getattr(value, "x", None), "route point x"),
        _require_real(getattr(value, "y", None), "route point y"),
        _require_real(getattr(value, "z", None), "route point z"),
    )


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _length(vector: tuple[float, float, float]) -> float:
    return sqrt(_dot(vector, vector))


def _normalized_route_progress(
    route: RaceRoute,
    *,
    initial_position: tuple[float, float, float],
    vehicle_position: tuple[float, float, float],
    next_checkpoint_index: int,
    finished: bool,
) -> float:
    if not isinstance(route, RaceRoute):
        raise ValueError("route must be RaceRoute")
    initial = _position_tuple(initial_position, "initial_position")
    vehicle = _position_tuple(vehicle_position, "vehicle_position")
    index = _require_int(
        next_checkpoint_index,
        "next_checkpoint_index",
        minimum=0,
    )
    finished = _require_bool(finished, "finished")
    if index > len(route.checkpoints):
        raise ValueError("next_checkpoint_index exceeds checkpoint count")
    if finished and index != len(route.checkpoints):
        raise ValueError("finished progress must complete all checkpoints")

    waypoints = (
        initial,
        *(_route_point(checkpoint.target_position) for checkpoint in route.checkpoints),
        _route_point(route.finish.target_position),
    )
    segment_vectors = tuple(
        _subtract(end, start)
        for start, end in zip(waypoints[:-1], waypoints[1:], strict=True)
    )
    segment_lengths = tuple(_length(vector) for vector in segment_vectors)
    total_length = sum(segment_lengths)
    if total_length <= 0.0 or not isfinite(total_length):
        raise ValueError("route reference length must be positive and finite")
    if finished:
        return 1.0

    completed_length = sum(segment_lengths[:index])
    segment_length = segment_lengths[index]
    if segment_length == 0.0:
        partial_length = 0.0
    else:
        segment_start = waypoints[index]
        offset = _subtract(vehicle, segment_start)
        projection = _dot(offset, segment_vectors[index]) / (segment_length * segment_length)
        fraction = min(1.0, max(0.0, projection))
        partial_length = fraction * segment_length

    normalized = (completed_length + partial_length) / total_length
    return min(1.0, max(0.0, normalized))


def summarize_track(
    episodes: tuple[EpisodeMetrics, ...],
) -> TrackEvaluationSummary:
    if not isinstance(episodes, tuple) or not episodes:
        raise ValueError("episodes must be a non-empty tuple")
    if not all(isinstance(episode, EpisodeMetrics) for episode in episodes):
        raise ValueError("episodes must contain EpisodeMetrics values")

    track_id = episodes[0].track_id
    if any(episode.track_id != track_id for episode in episodes[1:]):
        raise ValueError("all episodes must share the same track_id")

    successes = tuple(episode for episode in episodes if episode.success)
    return TrackEvaluationSummary(
        track_id=track_id,
        episode_count=len(episodes),
        success_count=len(successes),
        success_rate=len(successes) / len(episodes),
        mean_final_normalized_route_progress=fmean(
            episode.final_normalized_route_progress for episode in episodes
        ),
        mean_episode_return=fmean(episode.episode_return for episode in episodes),
        mean_collision_count=fmean(episode.collision_count for episode in episodes),
        mean_path_length=fmean(episode.path_length for episode in episodes),
        mean_speed=fmean(episode.mean_speed for episode in episodes),
        mean_airborne_steps=fmean(episode.airborne_steps for episode in episodes),
        mean_boost_applied_steps=fmean(
            episode.boost_applied_steps for episode in episodes
        ),
        mean_slow_zone_applied_steps=fmean(
            episode.slow_zone_applied_steps for episode in episodes
        ),
        mean_completion_time_for_successes=(
            fmean(
                episode.completion_time
                for episode in successes
                if episode.completion_time is not None
            )
            if successes
            else None
        ),
    )
