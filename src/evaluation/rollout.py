"""Read-only seeded policy rollouts shared by inference and exploratory evaluation."""

from collections.abc import Callable
from hashlib import sha256
import json

import numpy as np


def evaluate_policy_episode(
    env,
    policy,
    *,
    policy_name: str,
    episode: int,
    environment_seed: int,
    on_step: Callable[[int, tuple], None] | None = None,
) -> dict[str, object]:
    """Evaluate one already-created environment and return the stable rollout row.

    This API never invokes training. The output fields and invalid-transition
    handling are shared with the bounded exploratory pilot for compatibility.
    """
    rng = np.random.default_rng(environment_seed)
    trace = sha256()
    total, collisions, off_route, steps = 0.0, 0, 0, 0
    completed_checkpoint_ids: list[str] = []
    directed_crossings: list[str] = []
    info: dict[str, object] = {}
    error = None
    try:
        observation, _ = env.reset(seed=environment_seed)
        while True:
            action = (
                rng.uniform(env.action_space.low, env.action_space.high).astype(
                    np.float32
                )
                if policy_name == "random"
                else policy.predict(observation, deterministic=True)
            )
            transition = env.step(action)
            observation, reward, terminated, truncated, info = transition
            steps += 1
            if on_step is not None:
                on_step(steps, transition)
            total += float(reward)
            race = info.get("race") or {}
            completed_checkpoint_ids.extend(race.get("completed_checkpoint_ids", ()))
            directed_crossings.extend(race.get("directed_crossings", ()))
            collisions += len(info["telemetry"]["obstacle_hit_ids"])
            off_route += int(info.get("race", {}).get("terminal_failure") is True)
            trace.update(
                json.dumps(
                    dict(
                        action=action.tolist(),
                        observation=observation.tolist(),
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    ),
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            )
            if terminated or truncated:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    outcome = (
        getattr(env, "last_research_outcome", None)
        if error
        else info.get("research_outcome")
    ) or {}
    valid = error is None and outcome.get("valid_sample") is True
    return {
        "policy": policy_name,
        "episode": episode,
        "environment_seed": environment_seed,
        "valid_sample": valid,
        "success": valid and outcome.get("kind") == "successful_finish",
        "progress": min(
            1.0,
            info.get("route_highwater_meters", 0.0) / env._authored_route.total_meters,
        ),
        "completed_checkpoint_ids": completed_checkpoint_ids,
        "directed_crossings": directed_crossings,
        "collisions": collisions,
        "off_route_events": off_route,
        "duration_s": info.get("elapsed_time", 0.0),
        "steps": steps,
        "return": total,
        "outcome": outcome,
        "error": error,
        "trace_sha256": trace.hexdigest(),
    }
