"""Finite-sample planning utilities for the draft simulation-only factorial."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log, log1p, sqrt
from pathlib import Path

import numpy as np


def _beta_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    value = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        value *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= 3e-14:
            return value
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if (type(value) not in (int, float) or not isfinite(value)
            or type(degrees_of_freedom) is not int or degrees_of_freedom < 1):
        raise ValueError("finite t statistic and positive integer degrees of freedom required")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value > 0.0 else tail


def student_t_critical(degrees_of_freedom: int, two_sided_alpha: float) -> float:
    if (type(degrees_of_freedom) is not int or degrees_of_freedom < 1
            or type(two_sided_alpha) not in (int, float)
            or not isfinite(two_sided_alpha) or not 0.0 < two_sided_alpha < 1.0):
        raise ValueError("positive degrees of freedom and alpha in (0,1) required")
    target = 1.0 - two_sided_alpha / 2.0
    lower, upper = 0.0, 64.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if student_t_cdf(middle, degrees_of_freedom) < target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def estimate_paired_t_power(*, blocks: int, mde: float, contrast_sd: float,
                            family_alpha: float = 0.05, contrasts: int = 3,
                            simulations: int = 500_000, seed: int = 20_261_020) -> dict:
    """Estimate two-sided paired-t power under normal seed-block contrasts.

    Bonferroni alpha is conservative for the planned three-test Holm family.
    A seeded Monte Carlo integrates the exact finite-sample noncentral-t
    representation; the reported interval is simulation error only.
    """
    values = (mde, contrast_sd, family_alpha)
    if any(type(value) not in (int, float) or not isfinite(value) for value in values):
        raise ValueError("MDE, SD and family alpha must be finite numbers")
    if (type(blocks) is not int or blocks < 2 or mde <= 0.0 or contrast_sd <= 0.0
            or not 0.0 < family_alpha < 1.0 or type(contrasts) is not int or contrasts < 1
            or type(simulations) is not int or simulations < 10_000
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid finite-sample power design")
    alpha = family_alpha / contrasts
    critical = student_t_critical(blocks - 1, alpha)
    rng = np.random.default_rng(seed)
    chi_square = rng.chisquare(blocks - 1, size=simulations)
    normal = rng.normal(size=simulations)
    noncentrality = (mde / contrast_sd) * sqrt(blocks)
    statistic = (normal + noncentrality) / np.sqrt(chi_square / (blocks - 1))
    power = float(np.mean(np.abs(statistic) >= critical))
    standard_error = sqrt(power * (1.0 - power) / simulations)
    return {
        "method": "seeded_monte_carlo_noncentral_t_two_sided_bonferroni",
        "assumption": "independent_normal_paired_seed_block_contrasts",
        "blocks": blocks,
        "degrees_of_freedom": blocks - 1,
        "mde": float(mde),
        "contrast_sd": float(contrast_sd),
        "family_alpha": float(family_alpha),
        "contrasts": contrasts,
        "per_contrast_alpha": alpha,
        "critical_t": critical,
        "simulations": simulations,
        "seed": seed,
        "power_estimate": power,
        "monte_carlo_95_percent_interval": [
            max(0.0, power - 1.96 * standard_error),
            min(1.0, power + 1.96 * standard_error),
        ],
    }


def holm_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if (not isinstance(p_values, tuple) or not p_values
            or any(type(value) not in (int, float) or not isfinite(value)
                   or not 0.0 <= value <= 1.0 for value in p_values)):
        raise ValueError("a tuple of finite p-values in [0,1] is required")
    ordered = sorted(enumerate(p_values), key=lambda row: row[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(p_values) - rank) * value))
        adjusted[index] = running
    return tuple(adjusted)


@dataclass(frozen=True, slots=True)
class CellSummary:
    successes: int
    episodes: int
    invalid_episodes: int
    aborted: bool
    evaluation_roster_sha256: str
    candidate_id: str
    seed_bundle: tuple[int, int, int]
    run_manifest_sha256: str


def factorial_seed_block_contrasts(blocks: dict, *, repository_root: str | Path) -> list[dict]:
    """Validate strict A/B/C/D TRAIN summaries and calculate paired block effects.

    Hash strings are caller-supplied labels only; the result deliberately does
    not authenticate training or evaluation artifacts.
    """
    from .simulation_only_protocol import confirmation_seed_bundles, load_simulation_only_manifest

    manifest, _ = load_simulation_only_manifest(repository_root)
    expected_seeds = set(confirmation_seed_bundles(repository_root))
    if not isinstance(blocks, dict) or set(blocks) != expected_seeds:
        raise ValueError("exactly the 24 reserved confirmation seed bundles are required")
    tracks, strata = manifest["tracks"], manifest["strata"]
    expected_cells = {(track, stratum) for track in tracks for stratum in strata}
    candidates = ("A", "B", "C", "D")
    episode_count = manifest["statistics"]["episodes_per_track_stratum_per_model"]
    roster_digest = None
    all_run_digests = set()
    estimates = []
    for seed_bundle in sorted(expected_seeds):
        block = blocks[seed_bundle]
        if not isinstance(block, dict) or set(block) != set(candidates):
            raise ValueError("each confirmation seed block requires exactly A/B/C/D")
        candidate_rates = {}
        candidate_digests = {}
        for candidate in candidates:
            cells = block[candidate]
            if not isinstance(cells, dict) or set(cells) != expected_cells:
                raise ValueError("each candidate requires all eight registered TRAIN task cells")
            rates = []
            run_digest = None
            for key in sorted(expected_cells):
                cell = cells[key]
                if (not isinstance(cell, CellSummary) or type(cell.successes) is not int
                        or type(cell.episodes) is not int or type(cell.invalid_episodes) is not int
                        or type(cell.aborted) is not bool or cell.episodes != episode_count
                        or cell.invalid_episodes != 0 or cell.aborted
                        or not 0 <= cell.successes <= cell.episodes):
                    raise ValueError("invalid or aborted simulation-only task cell")
                if (cell.candidate_id != candidate or cell.seed_bundle != seed_bundle
                        or not _is_digest(cell.run_manifest_sha256)):
                    raise ValueError("candidate, seed bundle or run-manifest label mismatch")
                if run_digest is None:
                    run_digest = cell.run_manifest_sha256
                elif run_digest != cell.run_manifest_sha256:
                    raise ValueError("candidate cells do not share one run-manifest label")
                if not _is_digest(cell.evaluation_roster_sha256):
                    raise ValueError("evaluation roster label must be lowercase SHA-256")
                if roster_digest is None:
                    roster_digest = cell.evaluation_roster_sha256
                elif roster_digest != cell.evaluation_roster_sha256:
                    raise ValueError("paired candidates require the same evaluation roster label")
                rates.append(cell.successes / cell.episodes)
            if run_digest in all_run_digests:
                raise ValueError("run-manifest labels must be unique across candidate seed blocks")
            all_run_digests.add(run_digest)
            candidate_rates[candidate] = sum(rates) / len(rates)
            candidate_digests[candidate] = run_digest
        a, b, c, d = (candidate_rates[candidate] for candidate in candidates)
        estimates.append({
            "seed_bundle": list(seed_bundle),
            "candidate_finish_probability": candidate_rates,
            "candidate_run_manifest_sha256": candidate_digests,
            "contrasts": {
                "reward_main_effect": ((b-a)+(d-c))/2.0,
                "corridor_main_effect": ((c-a)+(d-b))/2.0,
                "interaction": (d-c)-(b-a),
            },
        })
    return estimates


def _is_digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def analyze_factorial_seed_blocks(estimates: list[dict], *, repository_root: str | Path) -> dict:
    """Proposed paired-t analysis; never treats hash labels as authenticated evidence."""
    from .simulation_only_protocol import confirmation_seed_bundles

    if not isinstance(estimates, list) or len(estimates) != 24:
        raise ValueError("exactly 24 paired confirmation seed-block estimates are required")
    expected_bundles = set(confirmation_seed_bundles(repository_root))
    names = ("reward_main_effect", "corridor_main_effect", "interaction")
    rows = []
    seed_bundles = set()
    run_digests = set()
    for row in estimates:
        if not isinstance(row, dict) or set(row) != {
                "seed_bundle", "candidate_finish_probability", "candidate_run_manifest_sha256", "contrasts"}:
            raise ValueError("invalid seed-block estimate fields")
        bundle = row["seed_bundle"]
        if (not isinstance(bundle, list) or len(bundle) != 3
                or any(type(seed) is not int or not 0 <= seed <= 2**32-1 for seed in bundle)
                or tuple(bundle) in seed_bundles):
            raise ValueError("malformed or duplicate seed-block identity")
        seed_bundles.add(tuple(bundle))
        probabilities = row["candidate_finish_probability"]
        hashes = row["candidate_run_manifest_sha256"]
        contrasts = row["contrasts"]
        if (not isinstance(probabilities, dict) or set(probabilities) != {"A", "B", "C", "D"}
                or not isinstance(hashes, dict) or set(hashes) != {"A", "B", "C", "D"}
                or not isinstance(contrasts, dict) or set(contrasts) != set(names)):
            raise ValueError("incomplete seed-block candidate summary")
        values = {}
        for candidate, probability in probabilities.items():
            if (type(probability) not in (int, float) or not isfinite(probability)
                    or not 0. <= probability <= 1.):
                raise ValueError("candidate finish probability outside [0,1]")
            digest = hashes[candidate]
            if not _is_digest(digest) or digest in run_digests:
                raise ValueError("missing, malformed or duplicate run-manifest hash label")
            run_digests.add(digest)
            values[candidate] = float(probability)
        a, b, c, d = (values[candidate] for candidate in ("A", "B", "C", "D"))
        expected = {
            "reward_main_effect": ((b-a)+(d-c))/2.0,
            "corridor_main_effect": ((c-a)+(d-b))/2.0,
            "interaction": (d-c)-(b-a),
        }
        if any(type(contrasts[name]) not in (int, float) or not isfinite(contrasts[name])
               or abs(contrasts[name]-expected[name]) > 1e-12 for name in names):
            raise ValueError("factorial contrast does not match candidate finish probabilities")
        rows.append((tuple(bundle), expected))
    if seed_bundles != expected_bundles:
        raise ValueError("analysis estimates must use the complete reserved confirmation roster")
    rows.sort(key=lambda item: item[0])
    alpha = 0.05
    raw_p = {}
    intervals = {}
    df = 23
    t_critical = student_t_critical(df, alpha)
    for name in names:
        values = np.asarray([row[name] for _, row in rows], dtype=np.float64)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        if sd == 0.0:
            raise ValueError("paired-t inference is undefined when seed-block contrast variance is zero")
        t_value = mean / (sd / sqrt(len(values)))
        p_value = min(1.0, 2.0 * (1.0 - student_t_cdf(abs(t_value), df)))
        raw_p[name] = p_value
        margin = t_critical * sd / sqrt(len(values))
        intervals[name] = [mean - margin, mean + margin]
    adjusted = holm_adjust(tuple(raw_p[name] for name in names))
    return {
        "status": "UNVERIFIED_SUMMARY_ONLY",
        "provenance_authenticated": False,
        "primary_test": "two_sided_paired_t",
        "primary_test_assumption": "independent_approximately_normal_PPO_seed_block_contrasts",
        "degrees_of_freedom": df,
        "confidence_intervals_95_percent": intervals,
        "raw_p_values": raw_p,
        "holm_adjusted_p_values": dict(zip(names, adjusted, strict=True)),
        "multiplicity": "holm_three_primary_contrasts_alpha_0.05",
    }
