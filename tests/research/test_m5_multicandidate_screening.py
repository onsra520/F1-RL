from scripts.m5_multicandidate_screening import (
    TRACKS,
    _episode_id,
    _rank_candidates,
    _seed_roster,
    normalized_trace_sha256,
)


def test_campaign_track_roster_uses_right_turn_v2_and_excludes_v1():
    assert TRACKS == (
        "straight-sprint-v1",
        "slalom-v1",
        "right-turn-train-v2",
        "ramp-v1",
        "left-turn-train-v2",
    )
    assert "right-turn-v1" not in " ".join(TRACKS)


def test_shared_seed_roster_is_reproducible_and_varied():
    first = _seed_roster()
    second = _seed_roster()
    assert first == second
    all_env = [row["environment_seed"] for values in first.values() for row in values]
    all_policy = [row["policy_seed"] for values in first.values() for row in values]
    assert len(set(all_env)) == 75
    assert len(set(all_policy)) == 75
    assert all(len(values) == 15 for values in first.values())


def test_episode_ids_are_unique_by_stage_and_mode():
    values = {
        _episode_id("stagea", "M4-C-seed0-original", "right-turn-train-v2", i, mode)
        for i in range(4)
        for mode in ("stochastic", "deterministic")
    }
    assert len(values) == 8
    assert all("right-turn-v1" not in value for value in values)


def test_normalized_trace_hash_ignores_worker_specific_case_id():
    first = [{"case_id": "case_w1", "step": 0, "action": [0.1, 0.8, 0.0]}]
    second = [{"case_id": "case_w8", "step": 0, "action": [0.1, 0.8, 0.0]}]
    assert normalized_trace_sha256(first) == normalized_trace_sha256(second)
    second[0]["action"] = [0.2, 0.8, 0.0]
    assert normalized_trace_sha256(first) != normalized_trace_sha256(second)


def test_candidate_rank_prioritizes_worst_track_reliability():
    track_names = TRACKS
    scorecard = {
        "pooled-high-catastrophic-track": {
            track: {"fixed_stochastic_10": {"finishes": 10 if track != track_names[0] else 0,
                                             "episodes": 10, "mean_progress": 1.0,
                                             "physical_crashes": 0, "step_limits": 0},
                    "fresh_stochastic_5": {"finishes": 5 if track != track_names[0] else 0,
                                            "episodes": 5, "mean_progress": 1.0,
                                            "physical_crashes": 0, "step_limits": 0}}
            for track in track_names
        },
        "balanced": {
            track: {"fixed_stochastic_10": {"finishes": 8, "episodes": 10,
                                             "mean_progress": 0.9, "physical_crashes": 1,
                                             "step_limits": 1},
                    "fresh_stochastic_5": {"finishes": 4, "episodes": 5,
                                            "mean_progress": 0.9, "physical_crashes": 1,
                                            "step_limits": 0}}
            for track in track_names
        },
    }

    ranked = _rank_candidates(scorecard)

    assert [row["candidate_id"] for row in ranked] == ["balanced", "pooled-high-catastrophic-track"]
