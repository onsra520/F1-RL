from __future__ import annotations

from collections.abc import Callable, Mapping
from math import dist, isclose, isfinite
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np

from ..episode import RaceRoute
from ..route_navigation import AuthoredRoute
from ..training.env_factory import F1RLEnvFactory
from .metrics import EpisodeMetrics, _normalized_route_progress


@runtime_checkable
class EvaluationPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray: ...


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _valid_terminal_failure(info: object) -> bool:
    if not isinstance(info, Mapping):
        return False
    outcome = info.get("research_outcome")
    race = info.get("race")
    return (
        isinstance(outcome, Mapping)
        and isinstance(race, Mapping)
        and outcome.get("version") == "wp7_episode_outcome_v1"
        and outcome.get("kind") == "valid_terminal_failure"
        and outcome.get("valid_sample") is True
        and outcome.get("reason") == "hard_route_violation"
        and race.get("terminal_failure") is True
    )


def _position(
    value: object,
    name: str = "vehicle_position",
) -> tuple[float, float, float]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-tuple")
    return (
        _real(value[0], f"{name}[0]"),
        _real(value[1], f"{name}[1]"),
        _real(value[2], f"{name}[2]"),
    )


def _ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return value


def _reset_position(info: object) -> tuple[float, float, float]:
    reset_info = _mapping(info, "reset info")
    telemetry = _mapping(reset_info.get("telemetry"), "reset telemetry")
    if "vehicle_position" not in telemetry:
        raise ValueError("reset telemetry requires vehicle_position")
    return _position(telemetry["vehicle_position"])


def _step_evidence(
    info: object,
    *,
    executed_steps: int,
    fixed_dt: float,
) -> tuple[
    int,
    float,
    int,
    bool,
    tuple[tuple[str, float], ...],
    tuple[float, float, float],
    float,
    bool,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    step_info = _mapping(info, "step info")

    required = (
        "tick",
        "elapsed_time",
        "next_checkpoint_index",
        "finished",
        "reward_components",
        "telemetry",
    )
    missing = tuple(name for name in required if name not in step_info)
    if missing:
        raise ValueError(f"step info is missing required fields: {missing!r}")

    tick = _integer(step_info["tick"], "tick")
    if tick != executed_steps:
        raise ValueError("tick must equal executed rollout step count")

    elapsed_time = _real(step_info["elapsed_time"], "elapsed_time", minimum=0.0)
    expected_elapsed = executed_steps * fixed_dt
    if not isclose(elapsed_time, expected_elapsed, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("elapsed_time must match executed steps * fixed_dt")

    next_checkpoint_index = _integer(
        step_info["next_checkpoint_index"],
        "next_checkpoint_index",
    )
    finished = _boolean(step_info["finished"], "finished")

    raw_components = _mapping(step_info["reward_components"], "reward_components")
    components: list[tuple[str, float]] = []
    for name, value in raw_components.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("reward component names must be non-empty strings")
        components.append((name, _real(value, f"reward component {name!r}")))

    telemetry = _mapping(step_info["telemetry"], "telemetry")
    telemetry_required = (
        "vehicle_position",
        "vehicle_speed",
        "grounded",
        "obstacle_hit_ids",
        "applied_boost_pad_ids",
        "applied_slow_zone_ids",
    )
    telemetry_missing = tuple(
        name for name in telemetry_required if name not in telemetry
    )
    if telemetry_missing:
        raise ValueError(
            f"telemetry is missing required fields: {telemetry_missing!r}"
        )

    vehicle_position = _position(telemetry["vehicle_position"])
    vehicle_speed = _real(
        telemetry["vehicle_speed"],
        "vehicle_speed",
        minimum=0.0,
    )
    grounded = _boolean(telemetry["grounded"], "grounded")
    obstacle_hit_ids = _ids(telemetry["obstacle_hit_ids"], "obstacle_hit_ids")
    boost_ids = _ids(
        telemetry["applied_boost_pad_ids"],
        "applied_boost_pad_ids",
    )
    slow_ids = _ids(
        telemetry["applied_slow_zone_ids"],
        "applied_slow_zone_ids",
    )

    return (
        tick,
        elapsed_time,
        next_checkpoint_index,
        finished,
        tuple(components),
        vehicle_position,
        vehicle_speed,
        grounded,
        obstacle_hit_ids,
        boost_ids,
        slow_ids,
    )


def evaluate_episode(
    *,
    track_id: str,
    factory: F1RLEnvFactory,
    policy: EvaluationPolicy,
    reset_seed: int | None = None,
    deterministic: bool = True,
    step_observer: Callable[[dict[str, object]], None] | None = None,
) -> EpisodeMetrics:
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError("track_id must be a non-empty string")
    if not isinstance(factory, F1RLEnvFactory):
        raise ValueError("factory must be F1RLEnvFactory")
    if not isinstance(policy, EvaluationPolicy) or not callable(
        getattr(policy, "predict", None)
    ):
        raise ValueError("policy must satisfy EvaluationPolicy")

    if reset_seed is not None and (type(reset_seed) is not int or not 0 <= reset_seed < 2**32):
        raise ValueError("reset_seed must be uint32 or None")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be bool")
    if step_observer is not None and not callable(step_observer):
        raise ValueError("step_observer must be callable or None")

    fixed_dt = float(factory.simulation_config.fixed_dt)
    route = RaceRoute.from_track(factory.track)
    authored_route = AuthoredRoute(factory.navigation) if factory.navigation is not None else None
    env = factory.create()
    try:
        observation, reset_info = env.reset() if reset_seed is None else env.reset(seed=reset_seed)
        previous_position = _reset_position(reset_info)

        executed_steps = 0
        episode_return = 0.0
        component_names: tuple[str, ...] | None = None
        component_totals: list[float] = []
        collision_count = 0
        path_length = 0.0
        speed_sum = 0.0
        max_speed = 0.0
        airborne_steps = 0
        boost_steps = 0
        slow_steps = 0

        final_elapsed = 0.0
        final_next_checkpoint_index = 0
        final_finished = False
        final_position = previous_position
        final_truncated = False
        route_highwater = 0.0

        while True:
            action = policy.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            executed_steps += 1

            reward_value = _real(reward, "reward")
            terminated = _boolean(terminated, "terminated")
            truncated = _boolean(truncated, "truncated")
            (
                _,
                elapsed_time,
                next_checkpoint_index,
                finished,
                components,
                vehicle_position,
                vehicle_speed,
                grounded,
                obstacle_ids,
                boost_ids,
                slow_ids,
            ) = _step_evidence(
                info,
                executed_steps=executed_steps,
                fixed_dt=fixed_dt,
            )
            if step_observer is not None:
                step_observer({
                    "step": executed_steps, "action": np.asarray(action).copy(),
                    "info": info, "reward": reward_value,
                    "terminated": terminated, "truncated": truncated,
                    "elapsed_time": elapsed_time,
                    "next_checkpoint_index": next_checkpoint_index,
                    "finished": finished,
                })
            if authored_route is not None:
                step_info = _mapping(info, "step info")
                if "route_highwater_meters" not in step_info:
                    raise ValueError("guided evaluator requires route_highwater_meters")
                step_highwater = _real(
                    step_info["route_highwater_meters"],
                    "route_highwater_meters",
                    minimum=0.0,
                )
                if step_highwater + 1e-9 < route_highwater:
                    raise ValueError("route_highwater_meters must be monotone")
                checkpoint_cap = (
                    authored_route.checkpoint_meters[next_checkpoint_index]
                    if next_checkpoint_index < len(authored_route.checkpoint_meters)
                    else authored_route.total_meters
                )
                if step_highwater > checkpoint_cap + 1e-9:
                    raise ValueError("route_highwater_meters exceeds unfinished checkpoint bound")
                route_highwater = max(route_highwater, step_highwater)

            names = tuple(name for name, _ in components)
            if component_names is None:
                component_names = names
                component_totals = [0.0] * len(names)
            elif names != component_names:
                raise ValueError(
                    "reward component names/order changed during evaluation episode"
                )
            for index, (_, value) in enumerate(components):
                component_totals[index] += value

            episode_return += reward_value
            path_length += dist(previous_position, vehicle_position)
            previous_position = vehicle_position
            collision_count += len(obstacle_ids)
            speed_sum += vehicle_speed
            max_speed = max(max_speed, vehicle_speed)
            airborne_steps += int(not grounded)
            boost_steps += int(bool(boost_ids))
            slow_steps += int(bool(slow_ids))

            final_elapsed = elapsed_time
            final_next_checkpoint_index = next_checkpoint_index
            final_finished = finished
            final_position = vehicle_position
            final_truncated = truncated

            if finished and not terminated:
                raise ValueError(
                    "terminal flags are inconsistent with finished race state"
                )
            if terminated or truncated:
                if finished:
                    if not terminated or truncated:
                        raise ValueError(
                            "terminal flags are inconsistent with successful episode"
                        )
                elif terminated:
                    if truncated or not _valid_terminal_failure(info):
                        raise ValueError(
                            "terminal flags are inconsistent with unsuccessful episode"
                        )
                elif not truncated:
                    raise ValueError(
                        "terminal flags are inconsistent with unsuccessful episode"
                    )
                break

        assert component_names is not None
        reward_component_totals = tuple(
            zip(component_names, component_totals, strict=True)
        )
        component_sum = sum(component_totals)
        if not isclose(
            episode_return,
            component_sum,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "episode reward does not match accumulated reward components"
            )

        normalized_progress = (
            (1.0 if final_finished else min(1.0, route_highwater / authored_route.total_meters))
            if authored_route is not None
            else _normalized_route_progress(
                route,
                initial_position=_reset_position(reset_info),
                vehicle_position=final_position,
                next_checkpoint_index=final_next_checkpoint_index,
                finished=final_finished,
            )
        )

        return EpisodeMetrics(
            track_id=track_id,
            success=final_finished,
            truncated=final_truncated,
            steps=executed_steps,
            episode_time=final_elapsed,
            completion_time=final_elapsed if final_finished else None,
            episode_return=episode_return,
            reward_component_totals=reward_component_totals,
            collision_count=collision_count,
            path_length=path_length,
            final_normalized_route_progress=normalized_progress,
            mean_speed=speed_sum / executed_steps,
            max_speed=max_speed,
            airborne_steps=airborne_steps,
            boost_applied_steps=boost_steps,
            slow_zone_applied_steps=slow_steps,
        )
    finally:
        env.close()
