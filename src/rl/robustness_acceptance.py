"""Exact preregistered per-policy, per-TRAIN-track reliability criterion.

One-sided Clopper-Pearson inverts a binomial test and does NOT assume
successes from separate tracks, stochastic seeds or different policies
are exchangeable. Multiple-testing adjustment, if required, is a distinct
protocol decision; never silently pool models or evaluate HELD-OUT.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, fsum, isfinite


def clopper_pearson_lower_bound(
    successes: int,
    episodes: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Compute exact one-sided confidence lower bound for Bernoulli success."""
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if type(successes) is not int or not 0 <= successes <= episodes:
        raise ValueError("successes must be an integer in [0, episodes]")
    if type(confidence) not in (float, int) or not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    if successes == 0:
        return 0.0
    alpha = 1.0 - float(confidence)
    if successes == episodes:
        return alpha ** (1.0 / episodes)

    # The binomial upper tail is monotone increasing in p. Its value
    # is alpha at the one-sided exact lower confidence limit.
    lower, upper = 0.0, 1.0
    for _ in range(75):
        p = (lower + upper) / 2.0
        tail = fsum(
            comb(episodes, k) * p**k * (1.0-p)**(episodes-k)
            for k in range(successes, episodes+1)
        )
        if tail < alpha:
            lower = p
        else:
            upper = p
    return (lower + upper) / 2.0


@dataclass(frozen=True, slots=True)
class TrackReliability:
    track_id: str
    policy_id: str
    condition: str
    successes: int
    episodes: int
    lower_bound: float
    accepted: bool


def evaluate_preregistered_track(
    *,
    track_id: str,
    policy_id: str,
    condition: str,
    successes: int,
    episodes: int,
    confidence: float = 0.95,
    lower_bound_threshold: float = 0.90,
    minimum_episodes: int = 60,
) -> TrackReliability:
    """Isolated per-track result, no held-out access or implicit aggregation."""
    for name, value in (("track_id", track_id), ("policy_id", policy_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if condition not in ("nominal", "randomized_train"):
        raise ValueError("condition must be a predeclared TRAIN evaluation stratum")
    if type(minimum_episodes) is not int or minimum_episodes <= 0:
        raise ValueError("minimum_episodes must be positive integer")
    if type(episodes) is not int or episodes < minimum_episodes:
        raise ValueError("each track/model/condition requires its prespecified evaluation budget")
    if type(lower_bound_threshold) not in (int, float) or not 0 < float(lower_bound_threshold) < 1:
        raise ValueError("invalid success lower-bound threshold")
    bound = clopper_pearson_lower_bound(successes, episodes, confidence=confidence)
    return TrackReliability(
        track_id, policy_id, condition, successes, episodes,
        bound, bound + 1e-12 >= float(lower_bound_threshold),
    )
