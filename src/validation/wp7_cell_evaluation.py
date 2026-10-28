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

from src.evaluation.evaluator import EvaluationPolicy, evaluate_episode
from src.training.env_factory import F1RLEnvFactory
from ..rl.action import ActionContract
from ..tracks.canonical import fingerprint_track_world
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
    return _spawn_ppo_cell(repository_root,payload,timeout_s=timeout_s,evidence_path=evidence_path)


def _spawn_ppo_cell(repository_root,payload,*,timeout_s,evidence_path):
    root=Path(repository_root).resolve(strict=True)
    worker=root/'scripts/wp7_ppo_fixture_worker.py'
    bootstrap='import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name="__main__")'
    try:
        with tempfile.TemporaryDirectory(prefix='wp7-ppo-worker-') as cache:
            completed=subprocess.run([sys.executable,'-I','-B','-X',f'pycache_prefix={cache}',
                '-c',bootstrap,str(root),str(worker)],input=pickle.dumps(payload),capture_output=True,
                timeout=timeout_s,cwd=root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('isolated PPO worker timed out') from exc
    if completed.returncode!=0:
        raise RuntimeError('isolated PPO worker failed: '+completed.stderr.decode(errors='replace')[-2000:])
    message=pickle.loads(completed.stdout)
    if message[0]!='ok':
        raise RuntimeError(f'isolated PPO worker failed: {message[1]}: {message[2]}')
    record=message[1]
    if record['sha256']!=digest({k:v for k,v in record.items() if k!='sha256'}):
        raise ValueError('isolated worker evidence checksum mismatch')
    validate_protocol(payload['protocol'],root)
    if evidence_path is not None:
        save_cell_evidence(record,evidence_path)
    return record


def _registered_cell_admission(root,protocol,policy_id,*,model_sha256,run_manifest_sha256,
                               run_manifest=None, review_envelope=None):
    validate_protocol(protocol,root)
    report=evaluate_admission(root,capture_runtime=True,profile=protocol_profile(protocol),
                              run_manifest=run_manifest, review_envelope=review_envelope)
    authorization=report['authorization']
    if (report['training_authorized'] is not True or authorization['verified'] is not True
            or report['owner_permissions']['training_permitted'] is not True):
        raise ScientificAdmissionError('research cell evaluation remains blocked',report)
    expected=dict(protocol_sha256=protocol['sha256'],policy_id=policy_id,
                  model_sha256=model_sha256,run_manifest_sha256=run_manifest_sha256)
    if any(value is None or authorization.get(key)!=value for key,value in expected.items()):
        raise ScientificAdmissionError('research cell run/model binding is not independently admitted',report)
    from .wp7_trusted_reviews import verify_repository_run_reviews
    try:
        verify_repository_run_reviews(root, authorization['run_review_envelope'],
            protocol_sha256=protocol['sha256'], run_manifest_sha256=run_manifest_sha256)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise ScientificAdmissionError('research cell requires authenticated run reviews',report) from exc
    # Authentic scientific reviews are explicitly non-authorizing. No owner
    # permit contract has been approved, so even a forged positive report must
    # not turn those signatures into execution permission.
    raise ScientificAdmissionError('authenticated training permit contract remains blocked', report)


def _registered_model_snapshot(*,protocol,policy_id,model_path,expected_model_sha256,
                               run_manifest,expected_run_manifest_sha256,artifact_directory):
    """Verify a run-bound immutable archive snapshot; this grants no permission."""
    from .wp7_run_manifest import verify_run_manifest
    from src.training.runner import _artifact_stem
    if not isinstance(artifact_directory,Path) or not isinstance(model_path,Path):
        raise ValueError('registered artifact directory and model Paths required')
    verify_run_manifest(run_manifest,artifact_directory,expected_sha256=expected_run_manifest_sha256)
    if run_manifest['protocol_sha256']!=protocol['sha256'] or run_manifest['receipt']['policy_id']!=policy_id:
        raise ValueError('model run protocol or policy identity mismatch')
    receipt=run_manifest['receipt']
    mode='multitrack' if len(receipt['training_track_ids'])>1 else 'singletrack'
    stem=_artifact_stem(mode,receipt['seed_bundle']['ppo'],receipt['sampling_seed'] if mode=='multitrack' else None)
    relative=model_path.resolve(strict=True).relative_to(artifact_directory.resolve(strict=True)).as_posix()
    if relative!=stem+'-final-model.zip':
        raise ValueError('registered primary final model selection differs')
    blob=_model_bytes(model_path)
    identity=dict(sha256=sha256(blob).hexdigest(),size_bytes=len(blob))
    entry=next((row for row in run_manifest['artifacts']['entries'] if row['path']==relative),None)
    if (identity['sha256']!=expected_model_sha256 or entry is None
            or any(entry[key]!=value for key,value in identity.items())):
        raise ValueError('registered model archive identity mismatch')
    return blob,{**identity,'run_manifest_sha256':expected_run_manifest_sha256,
                 'policy_id':policy_id,'artifact_path':relative}


def evaluate_registered_cell(
    repository_root: Path, protocol: dict, *, policy_id: str, track_id: str,
    stratum: str, model_path: Path,
    policy_for_seed: Callable[[int], EvaluationPolicy] | None = None,
    expected_model_sha256: str | None = None, run_manifest: dict | None = None,
    expected_run_manifest_sha256: str | None = None, artifact_directory: Path | None = None,
    review_envelope: dict | None = None,
    timeout_s: float = 3600., evidence_path: Path | None = None,
) -> dict:
    """Run-bound isolated inference; parent and child independently require admission.

    The legacy callable argument remains accepted for denied callers. Admitted
    execution must use immutable model bytes and cannot inject a policy callback.
    """
    root=Path(repository_root).resolve(strict=True)
    evaluation_seed(protocol,policy_id,track_id,stratum,0)
    _registered_cell_admission(root,protocol,policy_id,model_sha256=expected_model_sha256,
                               run_manifest_sha256=expected_run_manifest_sha256,
                               run_manifest=run_manifest, review_envelope=review_envelope)
    if policy_for_seed is not None:
        raise ValueError('registered inference forbids caller-supplied policy callbacks')
    if type(timeout_s) not in (int,float) or not 0.<timeout_s<=86400.:
        raise ValueError('registered worker timeout must be within (0,86400]')
    blob,model=_registered_model_snapshot(protocol=protocol,policy_id=policy_id,model_path=model_path,
        expected_model_sha256=expected_model_sha256,run_manifest=run_manifest,
        expected_run_manifest_sha256=expected_run_manifest_sha256,artifact_directory=artifact_directory)
    payload=dict(worker_kind='registered',protocol=protocol,policy_id=policy_id,track_id=track_id,stratum=stratum,
        model_bytes=blob,model=model,run_manifest=run_manifest,artifact_directory=str(artifact_directory),
        model_path=str(model_path),expected_run_manifest_sha256=expected_run_manifest_sha256,
        review_envelope=review_envelope,
        expected_model_sha256=expected_model_sha256,episodes=protocol['registration']['evaluation']['episodes_per_cell'],
        device=protocol['requested_device'])
    return _spawn_ppo_cell(root,payload,timeout_s=timeout_s,evidence_path=evidence_path)
