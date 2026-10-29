"""Deterministic WP3 evaluation roster and seed contract."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ._artifacts import _canonical


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _seed(domain: str, *parts: object) -> int:
    raw = "/".join([domain, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:4], "big")


def build_evaluation_roster(root: Path, campaign_manifest: dict, config: dict) -> dict:
    evaluation = config["evaluation"]
    binding = {
        "schema_version": evaluation["schema_version"],
        "campaign_manifest_sha256": campaign_manifest["sha256"],
        "source_commit": campaign_manifest["source_commit"],
        "simulation_only_config_sha256": campaign_manifest["simulation_only_config_sha256"],
        "registration_sha256": campaign_manifest["registration_sha256"],
        "track_manifest_sha256": campaign_manifest["track_manifest_sha256"],
        "evaluation": evaluation,
    }
    evaluation_config_sha256 = _digest(binding)
    seed_rows = []
    environment_seeds = set()
    policy_seeds = set()
    per_cell = evaluation["episodes_per_track_stratum_model"]
    schedule = campaign_manifest["execution_order"]
    for model in schedule:
        for track_id in evaluation["tracks"]:
            for stratum in evaluation["strata"]:
                episodes = []
                for episode in range(per_cell):
                    environment_seed = _seed(
                        "f1rl-wp3-eval-environment-seed-v1", evaluation_config_sha256,
                        track_id, stratum, episode,
                    )
                    policy_seed = _seed(
                        "f1rl-wp3-eval-policy-seed-v1", evaluation_config_sha256,
                        track_id, stratum, episode,
                    )
                    environment_seeds.add(environment_seed)
                    policy_seeds.add(policy_seed)
                    episodes.append({"episode": episode, "environment_seed": environment_seed,
                                     "policy_seed": policy_seed})
                seed_rows.append({"model_id": model["model_id"], "candidate_id": model["candidate_id"],
                                  "seed_index": model["seed_index"], "track_id": track_id,
                                  "stratum": stratum, "episodes": episodes})
    # Same cell/episode seeds are intentionally paired across all 12 models.
    for track_id in evaluation["tracks"]:
        for stratum in evaluation["strata"]:
            for episode in range(per_cell):
                env_values = {row["episodes"][episode]["environment_seed"] for row in seed_rows
                              if row["track_id"] == track_id and row["stratum"] == stratum}
                policy_values = {row["episodes"][episode]["policy_seed"] for row in seed_rows
                                 if row["track_id"] == track_id and row["stratum"] == stratum}
                if len(env_values) != 1 or len(policy_values) != 1:
                    raise ValueError("evaluation seeds must be identical across candidates and PPO seeds")
    expected_unique_seed_count = len(evaluation["tracks"]) * len(evaluation["strata"]) * per_cell
    if (len(environment_seeds) != expected_unique_seed_count
            or len(policy_seeds) != expected_unique_seed_count):
        raise ValueError("evaluation environment seed derivation collision")
    roster = {
        **binding, "evaluation_config_sha256": evaluation_config_sha256,
        "seed_rows": seed_rows,
        "expected_cells": len(seed_rows),
        "expected_episodes": len(seed_rows) * per_cell,
        "candidate_independent_paired_seeds": True,
        "scientific_evidence": False, "heldout_access": False,
    }
    roster["sha256"] = _digest(roster)
    return roster
