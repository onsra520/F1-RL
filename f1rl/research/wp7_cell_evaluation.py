"""Seeded stochastic evaluation cells; scientific admission remains closed."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
from typing import Callable

from ..learning.evaluation.evaluator import EvaluationPolicy, evaluate_episode
from ..learning.training.env_factory import F1RLEnvFactory
from .wp7_admission import evaluate_admission, ScientificAdmissionError
from .wp7_protocol import (
    digest, evaluation_seed, policy_stochastic_seed, validate_protocol,
)


def _model_identity(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.suffix.lower() != ".zip" or path.is_symlink() or not path.is_file():
        raise ValueError("model identity requires an existing regular ZIP path")
    return {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def _execute_cell(*, protocol: dict, policy_id: str, track_id: str, stratum: str,
                  model: dict, episodes: int,
                  factory_for_episode: Callable[[int], F1RLEnvFactory],
                  policy_for_seed: Callable[[int], EvaluationPolicy], scope: str) -> dict:
    if type(episodes) is not int or episodes < 1:
        raise ValueError("positive episode count required")
    record: dict = {
        "schema_version": "wp7_seeded_cell_execution_v1", "scope": scope,
        "protocol_sha256": protocol["sha256"], "policy_id": policy_id,
        "track_id": track_id, "stratum": stratum, "model": model,
        "scientific_acceptance": False, "scientific_evidence": False, "status": "running", "episodes": [],
    }
    for index in range(episodes):
        env_seed = evaluation_seed(protocol, policy_id, track_id, stratum, index)
        rng_seed = policy_stochastic_seed(protocol, policy_id, track_id, stratum, index)
        row = {"index": index, "environment_seed": env_seed, "policy_rng_seed": rng_seed}
        try:
            result = evaluate_episode(
                track_id=track_id, factory=factory_for_episode(index),
                policy=policy_for_seed(rng_seed),
                reset_seed=env_seed, deterministic=False,
            )
            row.update({"valid": True, "metrics": asdict(result)})
        except Exception as exc:
            row.update({"valid": False, "error_type": type(exc).__name__, "error": str(exc)})
            record["episodes"].append(row)
            record["status"] = "invalid_aborted"
            record["invalid_episode_index"] = index
            break
        record["episodes"].append(row)
    else:
        record["status"] = "completed"
    record["valid_episode_count"] = sum(row["valid"] for row in record["episodes"])
    record["sha256"] = digest(record)
    return record


def save_cell_evidence(record: dict, output_path: Path) -> None:
    """Persist invalid-aborted or completed cells; refuse overwrite."""
    if not isinstance(output_path, Path) or output_path.suffix.lower() != ".json":
        raise ValueError("cell evidence output must be a JSON Path")
    if record.get("sha256") != digest({k: v for k, v in record.items() if k != "sha256"}):
        raise ValueError("cell evidence checksum mismatch")
    if output_path.exists() or output_path.is_symlink() or not output_path.parent.is_dir():
        raise ValueError("cell evidence destination must be new in an existing directory")
    data = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".cell-", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with output_path.open("xb") as dest:
            dest.write(tmp.read_bytes())
            dest.flush()
            os.fsync(dest.fileno())
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def execute_software_fixture_cell(
    *, repository_root: Path, protocol: dict, policy_id: str, track_id: str,
    stratum: str, factory: F1RLEnvFactory, model_path: Path,
    policy_for_seed: Callable[[int], EvaluationPolicy], episodes: int = 2,
    evidence_path: Path | None = None,
) -> dict:
    """Execute software-only legacy Gym fixtures, NEVER WP7 research rollouts."""
    validate_protocol(protocol, repository_root)
    if not isinstance(factory, F1RLEnvFactory) or factory.requires_scientific_admission:
        raise ValueError("research physics forbidden in synthetic cell executor")
    if type(episodes) is not int or not 1 <= episodes <= 60:
        raise ValueError("fixture episodes must be within [1, 60]")
    record = _execute_cell(
        protocol=protocol, policy_id=policy_id, track_id=track_id, stratum=stratum,
        model=_model_identity(model_path), episodes=episodes,
        factory_for_episode=lambda index: factory, policy_for_seed=policy_for_seed,
        scope="synthetic_software_fixture_not_scientific_evidence",
    )
    if evidence_path is not None:
        save_cell_evidence(record, evidence_path)
    return record


def evaluate_registered_cell(
    repository_root: Path, protocol: dict, *, policy_id: str, track_id: str,
    stratum: str, model_path: Path,
    policy_for_seed: Callable[[int], EvaluationPolicy],
    evidence_path: Path | None = None,
) -> dict:
    """Research entrypoint requires independently verified approval and worker."""
    validate_protocol(protocol, repository_root)
    evaluation_seed(protocol, policy_id, track_id, stratum, 0)
    report = evaluate_admission(repository_root, capture_runtime=True)
    if (not report["training_authorized"]
            or not report["authorization"]["verified"]
            or not report["owner_permissions"]["training_permitted"]):
        raise ScientificAdmissionError("research cell evaluation remains blocked", report)
    raise RuntimeError("independently reviewed isolated PPO stochastic worker is not registered")
