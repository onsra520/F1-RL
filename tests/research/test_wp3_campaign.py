from __future__ import annotations

import json
import os

import pytest

from src.local_demo.campaign import (
    CAMPAIGN_CONFIG,
    _validate_training_contact_terminal,
    _expected_campaign_config,
    _iter_jsonl,
    _load_campaign_config,
    model_schedule,
    _windows_process_sample,
)
from src.local_demo.evaluation import build_evaluation_roster
from src.research.simulation_only_protocol import load_simulation_only_manifest


ROOT = __import__("pathlib").Path(__file__).parents[2]


def test_wp3_campaign_config_is_frozen_and_schedule_has_exact_seeded_order():
    config, _, _ = _load_campaign_config(ROOT)
    simulation, _ = load_simulation_only_manifest(ROOT)
    schedule = model_schedule(config, simulation)

    assert config == _expected_campaign_config()
    assert [(row["candidate_id"], row["seed_index"]) for row in schedule] == [
        ("C", 0), ("A", 0), ("B", 0), ("D", 0),
        ("A", 1), ("B", 1), ("C", 1), ("D", 1),
        ("A", 2), ("B", 2), ("C", 2), ("D", 2),
    ]
    assert len({row["run_relative_path"] for row in schedule}) == 12
    assert all(row["seed_bundle"]["training_reset"] == 1000 + row["seed_index"]
               for row in schedule)


def test_wp3_evaluation_roster_is_candidate_independent_paired_and_collision_checked():
    config, _, _ = _load_campaign_config(ROOT)
    simulation, _ = load_simulation_only_manifest(ROOT)
    schedule = model_schedule(config, simulation)
    campaign = {
        "sha256": "a" * 64,
        "source_commit": "b" * 40,
        "simulation_only_config_sha256": "c" * 64,
        "registration_sha256": "d" * 64,
        "track_manifest_sha256": "e" * 64,
        "execution_order": schedule,
    }

    roster = build_evaluation_roster(ROOT, campaign, config)

    assert roster["expected_cells"] == 96
    assert roster["expected_episodes"] == 5760
    rows = {(row["track_id"], row["stratum"], row["model_id"]): row for row in roster["seed_rows"]}
    first = rows[("straight-sprint-v1", "nominal", "C-seed0")]["episodes"][0]
    paired = rows[("straight-sprint-v1", "nominal", "D-seed2")]["episodes"][0]
    assert first == paired
    assert roster["candidate_independent_paired_seeds"] is True


def test_training_jsonl_parser_fails_closed_on_size_truncation_and_duplicate_keys(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"event":"step"}\n', encoding="utf-8")
    assert list(_iter_jsonl(path, 32)) == [{"event": "step"}]

    path.write_bytes(b'{"event":"step"}\n')
    with pytest.raises(ValueError, match="oversized"):
        list(_iter_jsonl(path, 8))

    path.write_bytes(b'{"event":"step"}')
    with pytest.raises(ValueError, match="incomplete"):
        list(_iter_jsonl(path, 32))

    path.write_bytes(b'{"event":"step","event":"other"}\n')
    with pytest.raises(ValueError, match="duplicate telemetry JSON key"):
        list(_iter_jsonl(path, 64))


def test_campaign_config_file_is_the_source_of_the_frozen_contract():
    config = json.loads((ROOT / CAMPAIGN_CONFIG).read_text(encoding="utf-8"))
    assert config == _expected_campaign_config()


@pytest.mark.parametrize(("code", "detail", "distance", "lateral_penalty", "demo_reward", "row_reward"), [
    ("unsupported_contact",
     "spatial body/contact sweep outside: invalid endpoint body or wheel support",
     2.2656793812716023, 1.0266606117438541,
     -26.026660611743853, -26.026660919189453),
    ("verified_obstacle_contact",
     "flight/substep body hit obstacle: no scored impact response registered",
     1.5758883456636512, 0.4966848155997039,
     -25.496684815599703, -25.496685028076172),
])
def test_campaign_accepts_captured_training_contact_terminal_contract(
    code, detail, distance, lateral_penalty, demo_reward, row_reward
):
    # Values are copied from actual C-seed0 attempt-v7 TRAIN telemetry rows.
    demo_contract = {
        "version": "straight_route_stability_demo_v2",
        "training_corridor_m": 3.0,
        "lateral_penalty_per_m2_s": 2.0,
        "terminal_penalty": 25.0,
    }
    row = {
        "terminated": True,
        "truncated": False,
        "termination_reason": detail,
        "reward": row_reward,
        "demo_training": {
            "version": "straight_route_stability_demo_v2",
            "training_corridor_m": 3.0,
            "registered_reward": 0.0,
            "lateral_distance_m": distance,
            "lateral_penalty": lateral_penalty,
            "terminal_penalty": 25.0,
            "reward": demo_reward,
            "curriculum_terminal": False,
            "physics_abort_terminal": True,
            "physics_failure_code": code,
            "physics_failure_detail": detail,
            "scientific_evidence": False,
        },
    }

    _validate_training_contact_terminal(row, demo_contract)


@pytest.mark.parametrize("mutation", [
    {"termination_reason": "invalid_physical_transition"},
    {"truncated": True},
    {"physics_failure_code": "unclassified_physics_error"},
    {"terminal_penalty": 24.0},
    {"reward": -25.0},
])
def test_campaign_rejects_inconsistent_training_contact_terminal(mutation):
    demo_contract = {
        "version": "straight_route_stability_demo_v2",
        "training_corridor_m": 3.0,
        "lateral_penalty_per_m2_s": 2.0,
        "terminal_penalty": 25.0,
    }
    row = {
        "terminated": True,
        "truncated": False,
        "termination_reason": "spatial body/contact sweep outside: invalid endpoint body or wheel support",
        "reward": -26.026660919189453,
        "demo_training": {
            "version": "straight_route_stability_demo_v2",
            "training_corridor_m": 3.0,
            "registered_reward": 0.0,
            "lateral_distance_m": 2.2656793812716023,
            "lateral_penalty": 1.0266606117438541,
            "terminal_penalty": 25.0,
            "reward": -26.026660611743853,
            "curriculum_terminal": False,
            "physics_abort_terminal": True,
            "physics_failure_code": "unsupported_contact",
            "physics_failure_detail": "spatial body/contact sweep outside: invalid endpoint body or wheel support",
            "scientific_evidence": False,
        },
    }
    for key, value in mutation.items():
        target = row if key in {"termination_reason", "truncated", "reward"} else row["demo_training"]
        target[key] = value

    with pytest.raises(ValueError, match="training contact terminal"):
        _validate_training_contact_terminal(row, demo_contract)


@pytest.mark.skipif(os.name != "nt", reason="WP3 campaign runs on local Windows")
def test_windows_resource_sample_reads_nonzero_process_working_set_and_cpu_time():
    sample = _windows_process_sample(os.getpid())

    assert sample is not None
    assert sample["rss_bytes"] > 0
    assert sample["cpu_seconds"] is not None
    assert sample["cpu_seconds"] >= 0
