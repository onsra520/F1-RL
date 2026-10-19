import pytest

from f1rl.research.wp2_statistics import (
    CellResult,
    confirmatory_factorial_contrasts,
    confirmatory_robustness_gate,
    paired_cluster_bootstrap_intervals,
    paired_seed_sample_size,
    paired_seed_t_intervals,
)


TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1")
STRATA = ("nominal", "randomized_train")
CELLS = tuple((track, stratum) for track in TRACKS for stratum in STRATA)


def _block(block_index, successes_by_candidate, *, roster="a" * 64):
    return {
        candidate: {
            cell: CellResult(successes, 60, 0, False, roster)
            for cell in CELLS
        }
        for candidate, successes in successes_by_candidate.items()
    }


def _confirmatory_blocks():
    rates = (30, 36, 42, 54)
    result = {}
    for index in range(12):
        delta = (0, 1, -1)[index % 3]
        candidate_successes = {
            "A": rates[0],
            "B": rates[1] + delta,
            "C": rates[2],
            "D": rates[3],
        }
        result[f"seed-block-{index:02d}"] = _block(index, candidate_successes)
    return result


def test_confirmatory_estimands_use_paired_seed_blocks_and_equal_task_cell_weights():
    blocks = _confirmatory_blocks()
    first = _block(0, {"A": 30, "B": 36, "C": 42, "D": 54})
    blocks["seed-block-00"] = first
    results = confirmatory_factorial_contrasts(blocks)

    assert len(results) == 12
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
    del blocks["seed-block-00"]["A"][CELLS[0]]
    with pytest.raises(ValueError, match="task-cell"):
        confirmatory_factorial_contrasts(blocks)

    blocks = _confirmatory_blocks()
    blocks["seed-block-00"]["D"][CELLS[0]] = CellResult(54, 60, 0, False, "b" * 64)
    with pytest.raises(ValueError, match="paired evaluation roster"):
        confirmatory_factorial_contrasts(blocks)


@pytest.mark.parametrize(
    ("invalid_episodes", "aborted"),
    [(1, False), (0, True)],
)
def test_confirmatory_contrasts_abort_invalid_or_aborted_cells(invalid_episodes, aborted):
    blocks = _confirmatory_blocks()
    blocks["seed-block-00"]["A"][CELLS[0]] = CellResult(
        30, 60, invalid_episodes, aborted, "a" * 64
    )
    with pytest.raises(ValueError, match="invalid or aborted"):
        confirmatory_factorial_contrasts(blocks)


def test_seed_cluster_bootstrap_is_reproducible_and_resamples_whole_blocks():
    estimates = confirmatory_factorial_contrasts(_confirmatory_blocks())
    intervals_a = paired_cluster_bootstrap_intervals(estimates, resamples=1000, seed=123)
    intervals_b = paired_cluster_bootstrap_intervals(estimates, resamples=1000, seed=123)

    assert intervals_a == intervals_b
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
                candidate_cells[cell_key] = CellResult(60, 60, 0, False, "a" * 64)
    result = confirmatory_robustness_gate(blocks)
    assert result["meets_preregistered_numeric_criterion"]
    assert len(result["lower_bounds"]) == 12 * 4 * 8
    assert not result["failing_cells"]
    assert all(value == pytest.approx(0.05 ** (1 / 60))
               for value in result["lower_bounds"].values())

    blocks["seed-block-00"]["A"][CELLS[0]] = CellResult(30, 60, 0, False, "a" * 64)
    result = confirmatory_robustness_gate(blocks)
    assert not result["meets_preregistered_numeric_criterion"]
    assert len(result["failing_cells"]) == 1
    assert result["failing_cells"][0] == ("seed-block-00", "A", *CELLS[0])


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
        blocks["seed-block-00"]["A"][CELLS[0]] = CellResult(30, 59, 0, False, "a" * 64)
    elif mutation == "success_count":
        blocks["seed-block-00"]["A"][CELLS[0]] = CellResult(61, 60, 0, False, "a" * 64)
    elif mutation == "roster_digest":
        blocks["seed-block-00"]["A"][CELLS[0]] = CellResult(30, 60, 0, False, "A" * 64)
    else:
        del blocks["seed-block-11"]
    with pytest.raises(ValueError, match=message):
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
