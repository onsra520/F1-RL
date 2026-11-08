"""Small offline helpers for the draft WP2 factorial analysis plan."""
from dataclasses import dataclass
from math import ceil, isclose, isfinite, sqrt
from pathlib import Path
from random import Random
from statistics import NormalDist


TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1")
STRATA = ("nominal", "randomized_train")
_CELLS = frozenset((track, stratum) for track in TRACKS for stratum in STRATA)
_CANDIDATES = ("A", "B", "C", "D")
_CONTRASTS = ("reward_main_effect", "corridor_main_effect", "interaction")
_BLOCK_COUNT = 12
_EPISODES_PER_CELL = 60
_ANALYSIS_BOOTSTRAP_RESAMPLES = 10_000
_ANALYSIS_BOOTSTRAP_SEED = 0
_T_CRITICAL_DF_11_95 = 2.200985160091638


@dataclass(frozen=True, slots=True)
class CellResult:
    """Unverified summary carrying caller-supplied candidate, seed and manifest labels."""

    successes: int
    episodes: int
    invalid_episodes: int
    aborted: bool
    evaluation_roster_sha256: str
    candidate_id: str
    seed_bundle: tuple[int, int, int]
    run_manifest_sha256: str


def _finite_number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _confirmation_seed_bundles():
    from ..exploratory.abcd import load_abcd_manifest

    manifest = load_abcd_manifest(Path(__file__).resolve().parents[2])
    protocol = manifest["statistical_protocol"]
    rule = protocol["confirmation_seed_bundle_rule"]
    return tuple((rule["ppo_start"] + index, rule["sampling_start"] + index,
                  rule["training_reset_start"] + index) for index in range(rule["count"]))


def _validated_estimates(estimates):
    if not isinstance(estimates, (tuple, list)) or len(estimates) != _BLOCK_COUNT:
        raise ValueError("exactly twelve paired seed-block estimates are required")
    rows = []
    seed_bundles = set()
    run_manifest_digests = set()
    expected_bundles = set(_confirmation_seed_bundles())
    for row in estimates:
        bundle = row.get("seed_bundle") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {
                "seed_bundle", "candidate_finish_probability", "candidate_run_manifest_sha256", "contrasts"}
                or not isinstance(bundle, list) or len(bundle) != 3
                or any(type(value) is not int or not 0 <= value <= 2**32 - 1 for value in bundle)
                or tuple(bundle) not in expected_bundles or tuple(bundle) in seed_bundles
                or not isinstance(row["candidate_finish_probability"], dict)
                or set(row["candidate_finish_probability"]) != set(_CANDIDATES)
                or not isinstance(row["candidate_run_manifest_sha256"], dict)
                or set(row["candidate_run_manifest_sha256"]) != set(_CANDIDATES)
                or not isinstance(row["contrasts"], dict)
                or set(row["contrasts"]) != set(_CONTRASTS)):
            raise ValueError("invalid, duplicate or unregistered paired confirmation seed-bundle estimate")
        seed_bundles.add(tuple(bundle))
        run_digests = tuple(row["candidate_run_manifest_sha256"][candidate] for candidate in _CANDIDATES)
        if (any(not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value) for value in run_digests)
                or len(set(run_digests)) != len(_CANDIDATES)
                or run_manifest_digests.intersection(run_digests)):
            raise ValueError("each candidate estimate requires its distinct run-manifest SHA-256")
        run_manifest_digests.update(run_digests)
        probabilities = {candidate: _finite_number(row["candidate_finish_probability"][candidate],
                                                    "candidate finish probability")
                         for candidate in _CANDIDATES}
        if any(not 0. <= value <= 1. for value in probabilities.values()):
            raise ValueError("candidate finish probabilities must be in [0, 1]")
        a, b, c, d = (probabilities[candidate] for candidate in _CANDIDATES)
        expected = {
            "reward_main_effect": ((b - a) + (d - c)) / 2,
            "corridor_main_effect": ((c - a) + (d - b)) / 2,
            "interaction": (d - c) - (b - a),
        }
        contrasts = {name: _finite_number(row["contrasts"][name], name) for name in _CONTRASTS}
        if any(not isclose(contrasts[name], value, rel_tol=0., abs_tol=1e-12)
               for name, value in expected.items()):
            raise ValueError("seed-block contrasts differ from the candidate finish probabilities")
        rows.append((tuple(bundle), contrasts))
    if seed_bundles != expected_bundles:
        raise ValueError("paired estimates must use the complete reserved confirmation seed bundles")
    return [contrasts for _, contrasts in sorted(rows)]


def confirmatory_factorial_contrasts(blocks):
    """Calculate proposed-design contrasts; caller-supplied run labels are not authenticated."""
    if not isinstance(blocks, dict) or len(blocks) != _BLOCK_COUNT:
        raise ValueError("exactly twelve independent confirmatory seed blocks are required")
    expected_bundles = set(_confirmation_seed_bundles())
    if set(blocks) != expected_bundles:
        raise ValueError("blocks must be keyed by the twelve reserved confirmation seed bundles")
    roster_sha256 = None
    estimates = []
    all_run_manifest_digests = set()
    for seed_bundle in sorted(expected_bundles):
        candidates = blocks[seed_bundle]
        if not isinstance(candidates, dict) or set(candidates) != set(_CANDIDATES):
            raise ValueError("each seed block must contain candidates A, B, C and D")
        probabilities = {}
        candidate_run_manifest_digests = {}
        run_manifest_digests = set()
        for candidate in _CANDIDATES:
            cells = candidates[candidate]
            if not isinstance(cells, dict) or set(cells) != _CELLS:
                raise ValueError("each candidate must contain the exact eight registered task-cell keys")
            rates = []
            run_manifest_sha256 = None
            for cell_key, cell in cells.items():
                if not isinstance(cell_key, tuple) or len(cell_key) != 2:
                    raise ValueError("task-cell keys must be (track, stratum) pairs")
                if (not isinstance(cell, CellResult)
                        or type(cell.successes) is not int
                        or type(cell.episodes) is not int
                        or type(cell.invalid_episodes) is not int
                        or type(cell.aborted) is not bool
                        or cell.episodes != _EPISODES_PER_CELL
                        or cell.invalid_episodes != 0
                        or cell.aborted
                        or not 0 <= cell.successes <= cell.episodes):
                    raise ValueError("confirmatory estimates reject invalid or aborted cells")
                if (cell.candidate_id != candidate or cell.seed_bundle != seed_bundle
                        or not isinstance(cell.run_manifest_sha256, str)
                        or len(cell.run_manifest_sha256) != 64
                        or any(char not in "0123456789abcdef" for char in cell.run_manifest_sha256)):
                    raise ValueError("cell provenance differs from its candidate, confirmation seeds or run manifest")
                if run_manifest_sha256 is None:
                    run_manifest_sha256 = cell.run_manifest_sha256
                elif cell.run_manifest_sha256 != run_manifest_sha256:
                    raise ValueError("candidate task cells must share one run-manifest identity")
                digest = cell.evaluation_roster_sha256
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise ValueError("evaluation roster identity must be a lowercase SHA-256 digest")
                if roster_sha256 is None:
                    roster_sha256 = digest
                elif digest != roster_sha256:
                    raise ValueError("all confirmatory cells require one paired evaluation roster")
                rates.append(cell.successes / cell.episodes)
            if run_manifest_sha256 in run_manifest_digests:
                raise ValueError("candidate run-manifest identities must be distinct within a seed block")
            if run_manifest_sha256 in all_run_manifest_digests:
                raise ValueError("candidate run-manifest identities must be unique across confirmation blocks")
            run_manifest_digests.add(run_manifest_sha256)
            all_run_manifest_digests.add(run_manifest_sha256)
            candidate_run_manifest_digests[candidate] = run_manifest_sha256
            probabilities[candidate] = sum(rates) / len(rates)
        a, b, c, d = (probabilities[candidate] for candidate in _CANDIDATES)
        estimates.append({
            "seed_bundle": list(seed_bundle),
            "candidate_finish_probability": probabilities,
            "candidate_run_manifest_sha256": candidate_run_manifest_digests,
            "contrasts": {
                "reward_main_effect": ((b - a) + (d - c)) / 2,
                "corridor_main_effect": ((c - a) + (d - b)) / 2,
                "interaction": (d - c) - (b - a),
            },
        })
    return estimates


def _percentile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def paired_cluster_bootstrap_intervals(estimates, *, resamples=10_000, seed=0, confidence=0.95):
    """Percentile intervals resampling complete paired seed blocks, never episodes."""
    rows = _validated_estimates(estimates)
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("bootstrap resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    confidence = _finite_number(confidence, "confidence")
    if not 0. < confidence < 1.:
        raise ValueError("confidence must be between zero and one")
    rng = Random(seed)
    tails = (1. - confidence) / 2.
    intervals = {}
    for contrast in _CONTRASTS:
        samples = []
        for _ in range(resamples):
            selected = [rows[rng.randrange(_BLOCK_COUNT)][contrast] for _ in range(_BLOCK_COUNT)]
            samples.append(sum(selected) / _BLOCK_COUNT)
        samples.sort()
        intervals[contrast] = (_percentile(samples, tails), _percentile(samples, 1. - tails))
    return intervals


def paired_seed_t_intervals(estimates, *, confidence=0.95):
    """Two-sided paired t sensitivity intervals for the preregistered 12 seed blocks."""
    rows = _validated_estimates(estimates)
    confidence = _finite_number(confidence, "confidence")
    if not 0. < confidence < 1.:
        raise ValueError("confidence must be between zero and one")
    if confidence != 0.95:
        raise ValueError("the preregistered t critical value is defined for 95% intervals")
    result = {}
    for contrast in _CONTRASTS:
        values = [row[contrast] for row in rows]
        mean = sum(values) / _BLOCK_COUNT
        variance = sum((value - mean) ** 2 for value in values) / (_BLOCK_COUNT - 1)
        margin = _T_CRITICAL_DF_11_95 * sqrt(variance / _BLOCK_COUNT)
        result[contrast] = (mean - margin, mean + margin)
    return result


def _paired_seed_block_sign_flip_p(values):
    observed = abs(sum(values) / len(values))
    extreme = 0
    allocations = 1 << len(values)
    for mask in range(allocations):
        signed_mean = sum(value if mask & (1 << index) else -value
                          for index, value in enumerate(values)) / len(values)
        if abs(signed_mean) >= observed - 1e-15:
            extreme += 1
    return extreme / allocations


def proposed_factorial_inference(estimates):
    """Return unverified draft inference; this helper cannot authenticate run evidence."""
    rows = _validated_estimates(estimates)
    bootstrap = paired_cluster_bootstrap_intervals(
        estimates, resamples=_ANALYSIS_BOOTSTRAP_RESAMPLES, seed=_ANALYSIS_BOOTSTRAP_SEED
    )
    t_sensitivity = paired_seed_t_intervals(estimates)
    raw = {contrast: _paired_seed_block_sign_flip_p([row[contrast] for row in rows])
           for contrast in _CONTRASTS}
    adjusted = holm_adjust(tuple(raw[contrast] for contrast in _CONTRASTS))
    return {
        "bootstrap_95_percent_intervals": bootstrap,
        "paired_t_sensitivity_95_percent_intervals": t_sensitivity,
        "raw_p_values": raw,
        "adjusted_p_values": dict(zip(_CONTRASTS, adjusted, strict=True)),
        "multiplicity": "holm_three_primary_contrasts_alpha_0.05",
        "primary_test": "two_sided_seed_block_sign_flip_4096",
        "primary_test_assumption": "independent_seed_blocks_and_sign_exchangeability_under_zero_effect",
        "bootstrap_resamples": _ANALYSIS_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": _ANALYSIS_BOOTSTRAP_SEED,
        "family_alpha": 0.05,
        "status": "UNVERIFIED_SUMMARY_ONLY",
        "provenance_authenticated": False,
    }


def confirmatory_robustness_gate(blocks):
    """Check the fixed one-sided CP threshold on every candidate/model task cell."""
    confirmatory_factorial_contrasts(blocks)
    from .wp7_robustness_gate import one_sided_cp_lower

    lower_bounds = {}
    for block_id in sorted(blocks):
        candidates = blocks[block_id]
        for candidate in _CANDIDATES:
            for track, stratum in sorted(_CELLS):
                cell = candidates[candidate][(track, stratum)]
                lower_bounds[(block_id, candidate, track, stratum)] = one_sided_cp_lower(
                    cell.successes, cell.episodes, alpha=0.05
                )
    failures = tuple(key for key, lower in lower_bounds.items() if lower < 0.90)
    return {
        "meets_preregistered_numeric_criterion": not failures,
        "required_lower_bound": 0.90,
        "one_sided_alpha": 0.05,
        "lower_bounds": lower_bounds,
        "failing_cells": failures,
    }


def paired_seed_sample_size(*, standard_deviation, minimum_meaningful_effect,
                            power=0.8, family_alpha=0.05, primary_contrasts=3):
    """Normal-approximation paired-block n with conservative two-sided Bonferroni alpha."""
    standard_deviation = _finite_number(standard_deviation, "standard deviation")
    effect = _finite_number(minimum_meaningful_effect, "minimum meaningful effect")
    power = _finite_number(power, "power")
    alpha = _finite_number(family_alpha, "family alpha")
    if (standard_deviation <= 0. or effect <= 0.
            or not 0.5 < power < 1.
            or not 0. < alpha < 1.
            or type(primary_contrasts) is not int or primary_contrasts <= 0):
        raise ValueError("usable positive SD/effect, power, alpha and contrast count are required")
    normal = NormalDist()
    critical = normal.inv_cdf(1. - alpha / (2 * primary_contrasts))
    detectable = normal.inv_cdf(power)
    required = ((critical + detectable) * standard_deviation / effect) ** 2
    if not isfinite(required):
        raise ValueError("sample-size calculation exceeds the finite numeric range")
    return max(2, ceil(required))


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in input order."""
    if not isinstance(p_values, (tuple, list)) or not p_values:
        raise ValueError('a nonempty p-value family is required')
    if any(type(p) not in (int, float) or not isfinite(p) or not 0. <= p <= 1.
           for p in p_values):
        raise ValueError('p-values must be finite numbers in [0, 1]')
    size = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda row: row[1])
    adjusted = [0.] * size
    running = 0.
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1., (size-rank)*value))
        adjusted[index] = running
    return tuple(adjusted)
