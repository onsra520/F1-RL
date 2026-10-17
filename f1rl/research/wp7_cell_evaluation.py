"""Seeded stochastic evaluation cells; scientific admission remains closed."""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
import pickle
import subprocess
import sys
from typing import Callable

from ..learning.evaluation.evaluator import EvaluationPolicy, evaluate_episode
from ..learning.training.env_factory import F1RLEnvFactory
from ..learning.action import ActionContract
from ..track_assets.canonical import fingerprint_track_world
from .wp7_admission import evaluate_admission, ScientificAdmissionError
from .wp7_protocol import (
    digest, evaluation_seed, policy_stochastic_seed, validate_protocol, protocol_profile,
)


def _model_bytes(path: Path) -> bytes:
    if not isinstance(path, Path) or path.suffix.lower() != ".zip" or path.is_symlink() or not path.is_file():
        raise ValueError("model identity requires an existing regular ZIP path")
    return path.read_bytes()


def _model_identity(path: Path) -> dict[str, object]:
    blob = _model_bytes(path)
    return {"sha256": sha256(blob).hexdigest(), "size_bytes": len(blob)}


def _fixture_identity(factory):
    def immutable_sets(value):
        if isinstance(value,(set,frozenset)):
            return sorted(value)
        raise TypeError(f"unsupported fixture configuration value: {type(value).__name__}")
    configuration = json.loads(json.dumps(asdict(factory),allow_nan=False,default=immutable_sets))
    def implementations(value):
        if is_dataclass(value):
            return {"type":f"{type(value).__module__}.{type(value).__qualname__}",
                    "fields":{f.name:implementations(getattr(value,f.name)) for f in fields(value)}}
        if isinstance(value,(tuple,list)):
            return [implementations(v) for v in value]
        if isinstance(value,dict):
            return {k:implementations(v) for k,v in value.items()}
        return None
    document = {"configuration":configuration,"world_sha256":fingerprint_track_world(factory.track),
                "action_contract":asdict(factory.action_contract or ActionContract()),
                "observation_features":list(factory.observation_encoder.feature_names),
                "implementations":implementations(factory),
                "reward_names":[term.name for term in factory.reward_composer.terms]}
    return {**document,"sha256":digest(document)}


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
        # Same-directory hard link publishes complete bytes without overwrite.
        os.link(tmp, output_path)
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
    if type(factory) is not F1RLEnvFactory or factory.requires_scientific_admission:
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


def execute_saved_ppo_fixture_cell(
    *, repository_root: Path, protocol: dict, policy_id: str, track_id: str, stratum: str,
    factory: F1RLEnvFactory, model_path: Path, expected_model_sha256: str,
    episodes: int = 2, device: str = "cpu", timeout_s: float = 45., evidence_path: Path | None = None,
) -> dict:
    """Run saved-model inference in a spawned process; never research rollouts.

    The immutable verified archive bytes cross the process boundary, eliminating
    a re-open race between hashing and model loading. This API cannot grant
    scientific admission and accepts only legacy software environment fixtures.
    """
    validate_protocol(protocol,repository_root)
    evaluation_seed(protocol,policy_id,track_id,stratum,0)
    if type(factory) is not F1RLEnvFactory or factory.requires_scientific_admission:
        raise ValueError("research physics forbidden in software worker")
    if type(episodes) is not int or not 1 <= episodes <= 60:
        raise ValueError("fixture episodes must be within [1,60]")
    if type(timeout_s) not in (int,float) or not 0. < timeout_s <= 60.:
        raise ValueError("worker timeout must be within (0,60]")
    if device not in ("cpu","cuda"):
        raise ValueError("explicit cpu or cuda device required")
    if device == "cuda":
        from .wp7_protocol import _runtime
        if not _runtime()["cuda_available"]:
            raise RuntimeError("explicit CUDA device unavailable")
    blob = _model_bytes(model_path)
    model = {"sha256":sha256(blob).hexdigest(),"size_bytes":len(blob)}
    if expected_model_sha256 != model["sha256"]:
        raise ValueError("model identity mismatch before worker start")
    payload = dict(protocol=protocol,policy_id=policy_id,track_id=track_id,stratum=stratum,
                   factory=factory,fixture=_fixture_identity(factory),model_bytes=blob,
                   model=model,episodes=episodes,device=device)
    root = Path(repository_root).resolve(strict=True)
    worker = root / "scripts/wp7_ppo_fixture_worker.py"
    try:
        completed = subprocess.run([sys.executable,"-B",str(worker)],input=pickle.dumps(payload),
            capture_output=True,timeout=timeout_s,cwd=root,
            env={**os.environ,"PYTHONPATH":str(root),"PYTHONDONTWRITEBYTECODE":"1"})
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("isolated PPO software worker timed out") from exc
    if completed.returncode != 0:
        raise RuntimeError("isolated PPO software worker failed: "+completed.stderr.decode(errors="replace")[-2000:])
    message = pickle.loads(completed.stdout)
    if message[0] != "ok":
        raise RuntimeError(f"isolated PPO software worker failed: {message[1]}: {message[2]}")
    record = message[1]
    if record["sha256"] != digest({k:v for k,v in record.items() if k != "sha256"}):
        raise ValueError("isolated worker evidence checksum mismatch")
    if evidence_path is not None:
        save_cell_evidence(record,evidence_path)
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
    report = evaluate_admission(repository_root, capture_runtime=True,profile=protocol_profile(protocol))
    if (not report["training_authorized"]
            or not report["authorization"]["verified"]
            or not report["owner_permissions"]["training_permitted"]):
        raise ScientificAdmissionError("research cell evaluation remains blocked", report)
    raise RuntimeError("independently reviewed isolated PPO stochastic worker is not registered")
