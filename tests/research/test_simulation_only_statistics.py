from hashlib import sha256
from pathlib import Path

import pytest

from f1rl.research.simulation_only_protocol import confirmation_seed_bundles, load_simulation_only_manifest
from f1rl.research.simulation_only_statistics import (
    CellSummary,
    analyze_factorial_seed_blocks,
    estimate_paired_t_power,
    factorial_seed_block_contrasts,
    student_t_cdf,
    student_t_critical,
)

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1")
STRATA = ("nominal", "randomized_train")
CELLS = tuple((track, stratum) for track in TRACKS for stratum in STRATA)


def _blocks():
    result = {}
    for index, bundle in enumerate(confirmation_seed_bundles(ROOT)):
        result[bundle] = {}
        for candidate, successes in zip(("A", "B", "C", "D"), (30, 36, 42, 54), strict=True):
            digest = sha256(f"{index}-{candidate}".encode()).hexdigest()
            result[bundle][candidate] = {
                key: CellSummary(successes, 60, 0, False, "a" * 64, candidate, bundle, digest)
                for key in CELLS
            }
    return result


def test_manifest_keeps_owner_selected_factorial_as_unapproved_draft():
    manifest, digest = load_simulation_only_manifest(ROOT)
    assert manifest["status"] == "DRAFT — NOT PREREGISTERED — NOT AUTHORIZED"
    assert manifest["statistics"]["confirmation_blocks"] == 24
    assert manifest["statistics"]["minimum_meaningful_effect"] == 0.15
    assert manifest["statistics"]["planning_contrast_sd"] == 0.20
    assert len(manifest["statistics"]["secondary_endpoints"]) == 6
    assert len(confirmation_seed_bundles(ROOT)) == 24
    assert len(digest) == 64


def test_paired_t_uses_finite_sample_critical_and_holm_family():
    estimates = factorial_seed_block_contrasts(_blocks(), repository_root=ROOT)
    result = analyze_factorial_seed_blocks(estimates, repository_root=ROOT)
    assert len(estimates) == 24
    assert result["primary_test"] == "two_sided_paired_t"
    assert result["degrees_of_freedom"] == 23
    assert result["status"] == "UNVERIFIED_SUMMARY_ONLY"
    assert result["provenance_authenticated"] is False
    assert result["primary_test_assumption"] == "independent_approximately_normal_PPO_seed_block_contrasts"
    assert set(result["holm_adjusted_p_values"]) == {
        "reward_main_effect", "corridor_main_effect", "interaction"
    }


def test_power_calculation_is_finite_sample_and_reproducible():
    first = estimate_paired_t_power(blocks=24, mde=0.15, contrast_sd=0.20, simulations=100_000)
    second = estimate_paired_t_power(blocks=24, mde=0.15, contrast_sd=0.20, simulations=100_000)
    assert first == second
    assert first["degrees_of_freedom"] == 23
    assert first["method"] == "seeded_monte_carlo_noncentral_t_two_sided_bonferroni"
    assert first["power_estimate"] >= 0.80
    assert first["monte_carlo_95_percent_interval"][0] >= 0.80
    assert student_t_critical(23, 0.05 / 3) == pytest.approx(2.5820171983, abs=1e-9)
    assert student_t_cdf(0.0, 23) == 0.5


def test_factorial_summary_rejects_unregistered_seed_or_bad_provenance():
    blocks = _blocks()
    replacement = blocks.pop(next(iter(blocks)))
    blocks[(0, 0, 1000)] = replacement
    with pytest.raises(ValueError, match="reserved confirmation"):
        factorial_seed_block_contrasts(blocks, repository_root=ROOT)

    blocks = _blocks()
    bundle = next(iter(blocks))
    old = blocks[bundle]["A"][CELLS[0]]
    blocks[bundle]["A"][CELLS[0]] = CellSummary(
        old.successes, old.episodes, old.invalid_episodes, old.aborted,
        old.evaluation_roster_sha256, "B", old.seed_bundle, old.run_manifest_sha256)
    with pytest.raises(ValueError, match="candidate, seed bundle"):
        factorial_seed_block_contrasts(blocks, repository_root=ROOT)


def test_factorial_summary_rejects_invalid_and_aborted_cells():
    blocks = _blocks()
    bundle = next(iter(blocks))
    old = blocks[bundle]["A"][CELLS[0]]
    blocks[bundle]["A"][CELLS[0]] = CellSummary(
        old.successes, old.episodes, 1, False, old.evaluation_roster_sha256,
        old.candidate_id, old.seed_bundle, old.run_manifest_sha256)
    with pytest.raises(ValueError, match="invalid or aborted"):
        factorial_seed_block_contrasts(blocks, repository_root=ROOT)
