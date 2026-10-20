from __future__ import annotations

import json

import pytest

from f1rl.local_demo.campaign import (
    CAMPAIGN_CONFIG,
    _expected_campaign_config,
    _iter_jsonl,
    _load_campaign_config,
    model_schedule,
)
from f1rl.local_demo.evaluation import build_evaluation_roster
from f1rl.research.simulation_only_protocol import load_simulation_only_manifest


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
