"""Strict reader for the versioned simulation-only A/B/C/D draft."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


MANIFEST_PATH = Path("configs/research/simulation-only-ppo-v1.json")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate simulation-only manifest field")
        result[key] = value
    return result


def load_simulation_only_manifest(root: str | Path) -> tuple[dict, str]:
    path = Path(root).resolve(strict=True) / MANIFEST_PATH
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    expected = {
        "schema_version", "status", "split", "claims_scope", "training", "candidates", "tracks",
        "strata", "seeds", "statistics", "evaluation", "approval_state",
    }
    if (not isinstance(value, dict) or set(value) != expected
            or value["schema_version"] != "f1rl_simulation_only_ppo_v1"
            or value["status"] != "DRAFT — NOT PREREGISTERED — NOT AUTHORIZED"
            or value["split"] != "TRAIN"
            or value["claims_scope"] != "simulation_only_no_physical_calibration_or_wp7_acceptance"
            or value["approval_state"] != "proposal_only"):
        raise ValueError("simulation-only manifest is malformed or scope-escalated")
    if value["training"] != {
        "timesteps_per_candidate_seed": 65536, "checkpoint_every_timesteps": 4096,
        "selection": "final_checkpoint_only", "max_wall_seconds": 7200,
        "device": "cpu", "n_envs": 1, "torch_threads": 8, "torch_interop_threads": 1,
    }:
        raise ValueError("simulation-only training limits changed without a new reviewed version")
    candidates = value["candidates"]
    if (not isinstance(candidates, list) or len(candidates) != 4
            or [row.get("candidate_id") for row in candidates] != ["A", "B", "C", "D"]
            or candidates != [
                {"candidate_id": "A", "lateral_penalty_per_m2_s": 2.0, "training_corridor_m": 2.0,
                 "demo_contract": "straight_route_stability_demo_v1"},
                {"candidate_id": "B", "lateral_penalty_per_m2_s": 4.0, "training_corridor_m": 2.0,
                 "demo_contract": "straight_route_stability_demo_v1"},
                {"candidate_id": "C", "lateral_penalty_per_m2_s": 2.0, "training_corridor_m": 3.0,
                 "demo_contract": "straight_route_stability_demo_v2"},
                {"candidate_id": "D", "lateral_penalty_per_m2_s": 4.0, "training_corridor_m": 3.0,
                 "demo_contract": "straight_route_stability_demo_v2"},
            ]):
        raise ValueError("simulation-only factorial candidates are not the owner-selected 2x2 design")
    if value["tracks"] != ["straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1"]:
        raise ValueError("simulation-only manifest must use the four registered TRAIN tracks")
    if value["strata"] != ["nominal", "randomized_train"]:
        raise ValueError("simulation-only manifest must use the two registered TRAIN strata")
    seeds = value["seeds"]
    if (not isinstance(seeds, dict) or set(seeds) != {"tuning", "confirmation_rule"}
            or seeds["tuning"] != [[0, 0, 1000], [1, 1, 1001], [2, 2, 1002]]
            or seeds["confirmation_rule"] != {
                "ppo_start": 100, "sampling_start": 200, "training_reset_start": 1100, "count": 24}):
        raise ValueError("simulation-only seed roster changed without a new reviewed version")
    statistics = value["statistics"]
    if (not isinstance(statistics, dict)
            or statistics.get("confirmation_blocks") != 24
            or statistics.get("minimum_meaningful_effect") != 0.15
            or statistics.get("planning_contrast_sd") != 0.2
            or statistics.get("target_power") != 0.8
            or statistics.get("family_alpha") != 0.05
            or statistics.get("primary_contrasts") != ["reward_main_effect", "corridor_main_effect", "interaction"]
            or statistics.get("secondary_endpoints") != [
                "mean_route_progress_fraction_at_termination", "mean_checkpoints_completed",
                "steering_rate_rms_rad_per_s", "collision_probability",
                "hard_route_failure_probability", "mean_episode_duration_seconds"]
            or statistics.get("multiplicity") != "holm_three_primary_contrasts"):
        raise ValueError("simulation-only D3 planning choices differ from owner-selected draft")
    if value["evaluation"] != {
            "task": "registered_original_task", "hard_route_boundary_m": 4.0,
            "semantics_unchanged": True,
            "seed_roster": "sha256_manifest_track_stratum_episode_candidate_independent"}:
        raise ValueError("simulation-only evaluation semantics or registered 4m boundary changed")
    return value, sha256(raw).hexdigest()


def confirmation_seed_bundles(root: str | Path) -> tuple[tuple[int, int, int], ...]:
    manifest, _ = load_simulation_only_manifest(root)
    rule = manifest["seeds"]["confirmation_rule"]
    return tuple((rule["ppo_start"] + index, rule["sampling_start"] + index,
                  rule["training_reset_start"] + index) for index in range(rule["count"]))
