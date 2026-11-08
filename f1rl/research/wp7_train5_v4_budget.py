"""Versioned, simulation-only TRAIN-5 checkpoint proposal. Never grants admission."""
from __future__ import annotations

from hashlib import sha256
import json
from math import ceil
from pathlib import Path

SCHEMA = "f1rl_wp7_train5_v4_local_checkpoint_draft_v1"
PINNED_PROFILE_SHA = "79172c7ca71b211d23047b234d03eb5f9537b599dac982f2913a61ce03942b1f"


class BudgetContractError(ValueError):
    pass


def load_v4_budget(root: Path) -> dict:
    filename = Path(root) / "configs/research/wp7-train5-v4-local-checkpoint-draft.json"
    return json.loads(filename.read_text(encoding="utf8"))


def validate_v4_budget(protocol: dict, pilot: dict) -> dict:
    """Validate a proposed budget against immutable legacy source, no PPO execution."""
    if not isinstance(protocol, dict) or not isinstance(pilot, dict):
        raise BudgetContractError("versioned protocol and pilot config required")
    if (protocol.get("schema") != SCHEMA or
        protocol.get("status") != "DRAFT_NOT_AUTHORIZED" or
        protocol.get("scope") != "SIMULATION_ONLY_TRAIN5_DEV_PROPOSAL" or
        protocol.get("source_baseline_commit") !=
            "a22bd65113a713d465b44d7b88d177243080aed1" or
        protocol.get("legacy_training_sha256_canonical_lf") != PINNED_PROFILE_SHA or
        protocol.get("legacy_training_profile") != "configs/recovery/wp7-pilot-200k.yml"):
        raise BudgetContractError("source or draft status mismatch")
    required_permissions = {
        "ppo_smoke", "ppo_training", "finetuning", "heldout", "full06", "scientific_release",
    }
    permissions = protocol.get("permissions")
    if (not isinstance(permissions, dict) or
        set(permissions) != required_permissions or
        any(permissions[k] is not False for k in required_permissions) or
        protocol.get("owner_scientific_signature") is not False or
        protocol.get("reviewer_approval") is not False or
        protocol.get("reserved_seed_consumption") != 0):
        raise BudgetContractError("draft cannot confer scientific admission")
    try:
        b = protocol["budget"]
        t = pilot["training"]
        ppo = pilot["ppo"]
        v = pilot["vector_env"]
        envs = v["n_envs"]
        n_steps = ppo["n_steps"]
        requested = t["total_timesteps"]
        batch = ppo["batch_size"]
        save_requested = t["checkpoint_every_steps"]
    except (KeyError, TypeError) as exc:
        raise BudgetContractError("training profile missing required settings") from exc
    if (type(envs) is not int or type(n_steps) is not int or
        type(requested) is not int or envs != 8 or n_steps != 256 or
        requested != 200000 or batch != 64 or save_requested != 25000):
        raise BudgetContractError("historical rollout settings must remain unchanged")
    quantum = envs * n_steps
    rollouts = ceil(requested / quantum)
    actual = rollouts * quantum
    expected = {
        "vector_envs": envs,
        "rollout_steps_per_env": n_steps,
        "rollout_quantum": quantum,
        "requested_steps": requested,
        "rollouts_per_policy": rollouts,
        "actual_final_checkpoint_step": actual,
        "overshoot_per_policy": actual - requested,
        "models": 6,
        "actual_total_steps": actual * 6,
        "checkpoint_strategy": "FINAL_ONLY_NO_INTERMEDIATE",
        "checkpoint_steps": [actual],
    }
    if type(b) is not dict or b != expected:
        raise BudgetContractError("rollout or checkpoint schedule deviates from registered proposal")
    return {
        "rollout_quantum": quantum,
        "rollouts_per_policy": rollouts,
        "actual_final_checkpoint_step": actual,
        "overshoot_per_policy": actual - requested,
        "total_actual_transitions": actual * 6,
        "checkpoint_steps": [actual],
        "training_permission": False,
        "heldout_permission": False,
    }


def verify_pinned_pilot_file(root: Path, protocol: dict) -> None:
    path = Path(root) / protocol["legacy_training_profile"]
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(canonical).hexdigest() != PINNED_PROFILE_SHA:
        raise BudgetContractError("legacy YAML profile differs from source-bound hash")
