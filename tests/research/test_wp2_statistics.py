from hashlib import sha256

import pytest

from f1rl.research.wp2_statistics import (
    CellResult,
    confirmatory_factorial_contrasts,
    confirmatory_robustness_gate,
    holm_adjust,
    paired_cluster_bootstrap_intervals,
    paired_seed_sample_size,
    paired_seed_t_intervals,
    proposed_factorial_inference,
    _paired_seed_block_sign_flip_p,
)


TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1")
STRATA = ("nominal", "randomized_train")
CELLS = tuple((track, stratum) for track in TRACKS for stratum in STRATA)


def _block(seed_bundle, successes_by_candidate, *, roster="a" * 64):
    return {
        candidate: {
            cell: CellResult(successes, 60, 0, False, roster, candidate, seed_bundle,
                             sha256(f"{seed_bundle}:{candidate}".encode()).hexdigest())
            for cell in CELLS
        }
        for candidate, successes in successes_by_candidate.items()
    }


def _confirmatory_blocks():
    rates = (30, 36, 42, 54)
    result = {}
    for index in range(12):
        seed_bundle = (100 + index, 200 + index, 1100 + index)
        delta = (0, 1, -1)[index % 3]
        candidate_successes = {
            "A": rates[0],
            "B": rates[1] + delta,
            "C": rates[2],
            "D": rates[3],
        }
        result[seed_bundle] = _block(seed_bundle, candidate_successes)
    return result


def test_confirmatory_estimands_use_paired_seed_blocks_and_equal_task_cell_weights():
    blocks = _confirmatory_blocks()
    first = _block((100, 200, 1100), {"A": 30, "B": 36, "C": 42, "D": 54})
    blocks[(100, 200, 1100)] = first
    results = confirmatory_factorial_contrasts(blocks)

    assert len(results) == 12
    assert results[0]["seed_bundle"] == [100, 200, 1100]
    assert results[0]["candidate_finish_probability"] == pytest.approx(
        {"A": 0.5, "B": 0.6, "C": 0.7, "D": 0.9}
    )
    assert results[0]["contrasts"] == pytest.approx({
        "reward_main_effect": 0.15,
        "corridor_main_effect": 0.25,
        "interaction": 0.10,
    })


def test_confirmatory_contrasts_reject_missing_cells_or_unpaired_evaluation_rosters():
    blocks = _confirmatory_blocks()
    del blocks[(100, 200, 1100)]["A"][CELLS[0]]
    with pytest.raises(ValueError, match="task-cell"):
        confirmatory_factorial_contrasts(blocks)

    blocks = _confirmatory_blocks()
    original = blocks[(100, 200, 1100)]["D"][CELLS[0]]
    blocks[(100, 200, 1100)]["D"][CELLS[0]] = CellResult(
        54, 60, 0, False, "b" * 64, original.candidate_id, original.seed_bundle,
        original.run_manifest_sha256
    )
    with pytest.raises(ValueError, match="paired evaluation roster"):
        confirmatory_factorial_contrasts(blocks)


@pytest.mark.parametrize(
    ("invalid_episodes", "aborted"),
    [(1, False), (0, True)],
)
def test_confirmatory_contrasts_abort_invalid_or_aborted_cells(invalid_episodes, aborted):
    blocks = _confirmatory_blocks()
    original = blocks[(100, 200, 1100)]["A"][CELLS[0]]
    blocks[(100, 200, 1100)]["A"][CELLS[0]] = CellResult(
        30, 60, invalid_episodes, aborted, "a" * 64, original.candidate_id,
        original.seed_bundle, original.run_manifest_sha256
    )
    with pytest.raises(ValueError, match="invalid or aborted"):
        confirmatory_factorial_contrasts(blocks)


def test_seed_cluster_bootstrap_is_reproducible_and_resamples_whole_blocks():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    intervals_a = paired_cluster_bootstrap_intervals(estimates, resamples=1000, seed=123)
    intervals_b = paired_cluster_bootstrap_intervals(estimates, resamples=1000, seed=123)
    intervals_reordered = paired_cluster_bootstrap_intervals(list(reversed(estimates)),
                                                              resamples=1000, seed=123)

    assert intervals_a == intervals_b
    assert intervals_a == intervals_reordered
    for name in ("reward_main_effect", "corridor_main_effect", "interaction"):
        lower, upper = intervals_a[name]
        mean = sum(row["contrasts"][name] for row in estimates) / 12
        assert lower <= mean <= upper
        assert lower < upper


def test_t_sensitivity_interval_uses_twelve_independent_seed_blocks():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    intervals = paired_seed_t_intervals(estimates)
    for name, expected_mean in (("reward_main_effect", 0.15), ("corridor_main_effect", 0.25),
                                ("interaction", 0.10)):
        lower, upper = intervals[name]
        assert lower < expected_mean < upper


def test_robustness_gate_checks_each_of_384_model_task_cells_without_pooling():
    blocks = _confirmatory_blocks()
    for candidates in blocks.values():
        for candidate_cells in candidates.values():
            for cell_key in CELLS:
                original = candidate_cells[cell_key]
                candidate_cells[cell_key] = CellResult(
                    60, 60, 0, False, "a" * 64, original.candidate_id,
                    original.seed_bundle, original.run_manifest_sha256
                )
    result = confirmatory_robustness_gate(blocks)
    assert result["meets_preregistered_numeric_criterion"]
    assert len(result["lower_bounds"]) == 12 * 4 * 8
    assert not result["failing_cells"]
    assert all(value == pytest.approx(0.05 ** (1 / 60))
               for value in result["lower_bounds"].values())

    original = blocks[(100, 200, 1100)]["A"][CELLS[0]]
    blocks[(100, 200, 1100)]["A"][CELLS[0]] = CellResult(
        30, 60, 0, False, "a" * 64, original.candidate_id,
        original.seed_bundle, original.run_manifest_sha256
    )
    result = confirmatory_robustness_gate(blocks)
    assert not result["meets_preregistered_numeric_criterion"]
    assert len(result["failing_cells"]) == 1
    assert result["failing_cells"][0] == ((100, 200, 1100), "A", *CELLS[0])


def test_paired_seed_sample_size_uses_explicit_mde_power_and_three_contrast_family():
    assert paired_seed_sample_size(
        standard_deviation=0.2,
        minimum_meaningful_effect=0.15,
        power=0.8,
        family_alpha=0.05,
        primary_contrasts=3,
    ) == 19


@pytest.mark.parametrize(
    "kwargs",
    [
        {"standard_deviation": 0.0, "minimum_meaningful_effect": 0.1},
        {"standard_deviation": 0.2, "minimum_meaningful_effect": 0.0},
        {"standard_deviation": 0.2, "minimum_meaningful_effect": 0.1, "power": 1.0},
    ],
)
def test_paired_seed_sample_size_rejects_unusable_planning_inputs(kwargs):
    with pytest.raises(ValueError):
        paired_seed_sample_size(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("episode_count", "invalid or aborted"),
        ("success_count", "invalid or aborted"),
        ("roster_digest", "lowercase SHA-256"),
        ("block_count", "twelve independent"),
    ],
)
def test_confirmatory_input_rejects_malformed_counts_roster_and_block_design(mutation, message):
    blocks = _confirmatory_blocks()
    if mutation == "episode_count":
        original = blocks[(100, 200, 1100)]["A"][CELLS[0]]
        blocks[(100, 200, 1100)]["A"][CELLS[0]] = CellResult(
            30, 59, 0, False, "a" * 64, original.candidate_id,
            original.seed_bundle, original.run_manifest_sha256
        )
    elif mutation == "success_count":
        original = blocks[(100, 200, 1100)]["A"][CELLS[0]]
        blocks[(100, 200, 1100)]["A"][CELLS[0]] = CellResult(
            61, 60, 0, False, "a" * 64, original.candidate_id,
            original.seed_bundle, original.run_manifest_sha256
        )
    elif mutation == "roster_digest":
        original = blocks[(100, 200, 1100)]["A"][CELLS[0]]
        blocks[(100, 200, 1100)]["A"][CELLS[0]] = CellResult(
            30, 60, 0, False, "A" * 64, original.candidate_id,
            original.seed_bundle, original.run_manifest_sha256
        )
    else:
        del blocks[(111, 211, 1111)]
    with pytest.raises(ValueError, match=message):
        confirmatory_factorial_contrasts(blocks)


@pytest.mark.parametrize("identity", ["candidate", "seed_bundle", "run_manifest", "inconsistent_run"])
def test_confirmatory_input_rejects_candidate_or_run_provenance_mismatch(identity):
    blocks = _confirmatory_blocks()
    bundle = (100, 200, 1100)
    candidate = blocks[bundle]["A"]
    cell_key = CELLS[0]
    original = candidate[cell_key]
    if identity == "candidate":
        update = {"candidate_id": "B"}
    elif identity == "seed_bundle":
        update = {"seed_bundle": (101, 201, 1101)}
    elif identity == "run_manifest":
        update = {"run_manifest_sha256": "NOT-A-DIGEST"}
    else:
        update = {"run_manifest_sha256": "f" * 64}
    candidate[cell_key] = CellResult(
        original.successes, original.episodes, original.invalid_episodes, original.aborted,
        original.evaluation_roster_sha256, update.get("candidate_id", original.candidate_id),
        update.get("seed_bundle", original.seed_bundle),
        update.get("run_manifest_sha256", original.run_manifest_sha256)
    )
    with pytest.raises(ValueError, match="provenance|run-manifest"):
        confirmatory_factorial_contrasts(blocks)


def test_interval_helpers_reject_invalid_confidence_and_sample_size_family():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    with pytest.raises(ValueError, match="confidence"):
        paired_cluster_bootstrap_intervals(estimates, confidence=1.0)
    with pytest.raises(ValueError, match="resamples"):
        paired_cluster_bootstrap_intervals(estimates, resamples=0)
    with pytest.raises(ValueError, match="95%"):
        paired_seed_t_intervals(estimates, confidence=0.9)
    with pytest.raises(ValueError, match="contrast count"):
        paired_seed_sample_size(standard_deviation=0.2, minimum_meaningful_effect=0.1,
                                primary_contrasts=True)


def test_interval_helpers_reject_forged_or_malformed_seed_block_estimates():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    malformed = [dict(row) for row in estimates]
    malformed[0] = {**malformed[0], "candidate_finish_probability": {"A": 0.5}}
    with pytest.raises(ValueError, match="estimate"):
        paired_seed_t_intervals(malformed)

    malformed = [dict(row) for row in estimates]
    malformed[0] = {**malformed[0], "contrasts": {**malformed[0]["contrasts"],
                                                    "interaction": 0.5}}
    with pytest.raises(ValueError, match="differ from"):
        paired_cluster_bootstrap_intervals(malformed, resamples=10)

    malformed = [dict(row) for row in estimates]
    bad_run_hashes = dict(malformed[0]["candidate_run_manifest_sha256"])
    bad_run_hashes["A"] = "bad"
    malformed[0] = {**malformed[0], "candidate_run_manifest_sha256": bad_run_hashes}
    with pytest.raises(ValueError, match="run-manifest"):
        paired_seed_t_intervals(malformed)


def test_confirmatory_design_rejects_tuning_and_arbitrary_seed_blocks():
    blocks = _confirmatory_blocks()
    blocks[(0, 0, 1000)] = blocks.pop((111, 211, 1111))
    with pytest.raises(ValueError, match="reserved confirmation seed bundles"):
        confirmatory_factorial_contrasts(blocks)

    blocks = _confirmatory_blocks()
    blocks["seed-block-00"] = blocks.pop((100, 200, 1100))
    with pytest.raises(ValueError, match="reserved confirmation seed bundles"):
        confirmatory_factorial_contrasts(blocks)


def test_draft_analysis_applies_holm_to_seed_block_sign_flip_tests():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    result = proposed_factorial_inference(estimates)
    assert set(result["raw_p_values"]) == {
        "reward_main_effect", "corridor_main_effect", "interaction"
    }
    assert all(0. < value <= 1. for value in result["raw_p_values"].values())
    assert result == proposed_factorial_inference(estimates)
    expected = holm_adjust(tuple(result["raw_p_values"][name] for name in
                                  ("reward_main_effect", "corridor_main_effect", "interaction")))
    assert tuple(result["adjusted_p_values"][name] for name in
                 ("reward_main_effect", "corridor_main_effect", "interaction")) == pytest.approx(expected)
    assert result["multiplicity"] == "holm_three_primary_contrasts_alpha_0.05"
    assert result["primary_test"] == "two_sided_seed_block_sign_flip_4096"
    assert result["primary_test_assumption"] == (
        "independent_seed_blocks_and_sign_exchangeability_under_zero_effect"
    )
    assert result["bootstrap_resamples"] == 10_000
    assert result["bootstrap_seed"] == 0
    assert result["status"] == "UNVERIFIED_SUMMARY_ONLY"
    assert result["provenance_authenticated"] is False


def test_sign_flip_reference_values_cover_zero_and_extreme_effects():
    assert _paired_seed_block_sign_flip_p([0.0] * 12) == 1.0
    assert _paired_seed_block_sign_flip_p([1.0] * 12) == 2 / 4096


def test_syntactically_valid_but_fabricated_run_hash_stays_unverified():
    blocks = _confirmatory_blocks()
    fabricated = "f" * 64
    for cell_key, original in blocks[(100, 200, 1100)]["A"].items():
        blocks[(100, 200, 1100)]["A"][cell_key] = CellResult(
            original.successes, original.episodes, original.invalid_episodes, original.aborted,
            original.evaluation_roster_sha256, original.candidate_id, original.seed_bundle, fabricated
        )
    analysis = proposed_factorial_inference(confirmatory_factorial_contrasts(blocks))
    assert analysis["status"] == "UNVERIFIED_SUMMARY_ONLY"
    assert analysis["provenance_authenticated"] is False
