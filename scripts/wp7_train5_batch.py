"""Prepare a bounded, unsigned internal TRAIN-5 batch for owner review."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import argparse
import base64
import getpass
import json
import os
import secrets
import subprocess

from f1rl.research.wp7_train5_v4_admission import (
    development_policy_seeds, development_seeds, load_v4_protocol, validate_v4_protocol,
)
from f1rl.research.wp7_train5_v4_checkpoint import FROZEN_CHECKPOINTS
from scripts.wp7_train5_v4_review_manifest import load_review_manifest


def build_plan(root, output, evidence_root, *, valid_hours):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from f1rl.research.wp7_train5_internal_execution import OWNER_KEY, _owner_public_key

    root = Path(root).resolve(strict=True)
    output = Path(output).resolve()
    if output == root or root in output.parents or output.exists():
        raise ValueError("output must be a new directory outside the source checkout")
    if type(valid_hours) is not int or not 1 <= valid_hours <= 168:
        raise ValueError("authorization validity must be between 1 and 168 hours")
    protocol, protocol_sha = load_v4_protocol(root)
    validate_v4_protocol(root, protocol)
    manifest, manifest_sha = load_review_manifest(root)
    public = _owner_public_key(root, manifest["review_source_bindings"][OWNER_KEY])
    openssh = Ed25519PublicKey.from_public_bytes(public).public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    fingerprint = "SHA256:" + base64.b64encode(sha256(base64.b64decode(openssh.split()[1])).digest()).decode().rstrip("=")
    env_seeds = development_seeds(root, protocol, protocol_sha)
    policy_seeds = development_policy_seeds(root, protocol, protocol_sha)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return {
        "schema": "train5_internal_batch_plan_v1", "authorized": False,
        "source_commit": head, "protocol_sha256": protocol_sha, "manifest_sha256": manifest_sha,
        "owner_key_fingerprint": fingerprint,
        "output_root": str(output), "evidence_root": str(Path(evidence_root).resolve()),
        "valid_hours": valid_hours, "scope": "SIMULATION_ONLY_FIVE_TRAIN_TRACKS",
        "runs": [{**run, "initial_checkpoint_sha256": FROZEN_CHECKPOINTS.get(run["initial_checkpoint"])}
                 for run in protocol["runs"]],
        "budget": protocol["budget"], "track_ids": protocol["track_ids"],
        "evaluation": {**protocol["development"], "environment_seed_count": len(env_seeds),
                       "policy_seed_count": len(policy_seeds)},
        "disclosures": ["15 historical job logs unavailable", "historical C2 replay unavailable",
                        "new seed global non-use not established", "independent scientific review not established"],
        "heldout_authorized": False,
    }


def _require_clean_source(root, head):
    current = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout
    if current != head or dirty:
        raise ValueError("source must be clean and match the reviewed commit")


def sign_plan(root, plan, private_key, *, now=None):
    """Produce per-run signed requests; caller must authenticate the owner key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from f1rl.research.wp7_train5_internal_execution import OWNER_KEY, _owner_public_key
    from scripts.wp7_train5_execution_authorization import authorization_message

    if not isinstance(plan, dict):
        raise ValueError("batch plan required")
    expected = build_plan(root, plan["output_root"], plan["evidence_root"], valid_hours=plan["valid_hours"])
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(plan) != canonical(expected):
        raise ValueError("batch plan differs from current registered scope")
    _require_clean_source(root, plan["source_commit"])
    manifest, _ = load_review_manifest(root)
    public = _owner_public_key(root, manifest["review_source_bindings"][OWNER_KEY])
    if not isinstance(private_key, Ed25519PrivateKey) or private_key.public_key().public_bytes_raw() != public:
        raise ValueError("signing key does not match the enrolled owner public key")
    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.utcoffset() is None or now.utcoffset().total_seconds() != 0:
        raise ValueError("UTC signing clock required")
    expires = now + timedelta(hours=plan["valid_hours"])
    requests = []
    for run in plan["runs"]:
        payload = {"schema": "train5_local_execution_review_v1", "purpose": "simulation_only_train5",
                   "source_commit": plan["source_commit"],
                   "protocol_sha256": plan["protocol_sha256"], "manifest_sha256": plan["manifest_sha256"],
                   "cohort": run["cohort"], "replicate": run["replicate"],
                   "output_root": plan["output_root"], "nonce": secrets.token_hex(16),
                   "issued_at_utc": now.isoformat(), "expires_at_utc": expires.isoformat(),
                   "accept_missing_seed_history": True, "accept_missing_c2_replay": True,
                   "accept_internal_review_pending": True}
        requests.append({"payload": payload, "signature_hex": private_key.sign(authorization_message(payload)).hex()})
    if len({row["payload"]["nonce"] for row in requests}) != 6:
        raise ValueError("duplicate authorization nonce")
    return requests


def preflight_checkpoints(root, evidence_root):
    """Verify real checkpoint policy tensors against an inert shape-only model."""
    import gymnasium as gym
    import numpy as np
    import random
    import torch
    import yaml
    from stable_baselines3.common.vec_env import DummyVecEnv
    from f1rl.learning.algorithms.ppo import PPOConfig, PPOModel
    from f1rl.research.wp7_train5_v4_runner import initialize_policy_from_archive
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories

    root = Path(root).resolve(strict=True)
    scenarios = preview_registered_factories(root, Path(evidence_root), cohort="train5_from_frozen_checkpoint", replicate=0)

    class ShapeOnly(gym.Env):
        requires_scientific_admission = False
        observation_space = scenarios.observation_space
        action_space = scenarios.action_space

        def reset(self, **kwargs):
            raise AssertionError("checkpoint preflight must not reset")

        def step(self, action):
            raise AssertionError("checkpoint preflight must not step")

    pilot = yaml.safe_load((root / "configs/recovery/wp7-pilot-200k.yml").read_text(encoding="utf8"))
    random_state, numpy_state = random.getstate(), np.random.get_state()
    env = DummyVecEnv([ShapeOnly])
    try:
        with torch.random.fork_rng(devices=[]):
            model = PPOModel.create(env, PPOConfig(**{**pilot["ppo"], "seed": None}))
            checked = {}
            for identity, expected in FROZEN_CHECKPOINTS.items():
                path = root / "models" / identity / "model-final.zip"
                data = path.read_bytes()
                actual = sha256(data).hexdigest()
                if actual != expected:
                    raise ValueError(f"frozen checkpoint differs: {identity}")
                initialize_policy_from_archive(model, data)
                checked[identity] = actual
            return checked
    finally:
        env.close()
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def execute_batch(root, plan, envelopes):
    """Run exactly the signed roster once; keep a durable non-resumable journal."""
    from f1rl.research.wp7_train5_internal_execution import verify_run
    from f1rl.research.wp7_train5_v4_checkpoint import prepare_v4_registered_runner
    from scripts.wp7_train5_v4_factory_preview import preview_registered_factories

    root = Path(root).resolve(strict=True)
    expected = build_plan(root, plan["output_root"], plan["evidence_root"], valid_hours=plan["valid_hours"])
    if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("batch plan differs from current registered scope")
    _require_clean_source(root, plan["source_commit"])
    if not isinstance(envelopes, list) or len(envelopes) != 6:
        raise ValueError("exactly six signed requests required")
    if len({row.get("payload", {}).get("nonce") for row in envelopes}) != 6:
        raise ValueError("six distinct authorization nonces required")
    output = Path(plan["output_root"])
    for run, envelope in zip(plan["runs"], envelopes, strict=True):
        verify_run(root, output, run["cohort"], run["replicate"], envelope)
    scenarios = [preview_registered_factories(root, Path(plan["evidence_root"]),
                 cohort=run["cohort"], replicate=run["replicate"]) for run in plan["runs"]]

    def originals():
        result = {name: sha256((root / "models" / name / "model-final.zip").read_bytes()).hexdigest()
                  for name in FROZEN_CHECKPOINTS}
        if result != FROZEN_CHECKPOINTS:
            raise ValueError("frozen checkpoint hash changed")
        return result

    before = originals()
    output.mkdir(parents=True, exist_ok=False)
    journal = output / "batch-journal.jsonl"

    def record(event):
        with journal.open("a", encoding="utf8", newline="\n") as handle:
            handle.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **event},
                                    sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    completed = []
    for run, envelope, scenario in zip(plan["runs"], envelopes, scenarios, strict=True):
        identity = {"cohort": run["cohort"], "replicate": run["replicate"]}
        record({"event": "started", **identity})
        print(f"Starting {run['cohort']} replicate {run['replicate']} (200704 actual steps)", flush=True)
        try:
            runner = prepare_v4_registered_runner(root, output, **identity)
            result = runner.run_authorized_multi(scenario, envelope=envelope, evidence_root=Path(plan["evidence_root"]))
            if originals() != before:
                raise ValueError("frozen checkpoint changed after run")
            result_path = result.final_model_path.resolve(strict=True)
            if not result_path.is_relative_to(output.resolve()):
                raise ValueError("final checkpoint escaped batch output")
            entry = {**identity, "run_directory": str(result.run_directory.resolve()),
                     "final_model": str(result_path), "actual_steps": result.actual_total_timesteps,
                     "sha256": sha256(result_path.read_bytes()).hexdigest()}
            record({"event": "completed", **entry})
            completed.append(entry)
            print(f"Completed {run['cohort']} replicate {run['replicate']}: {result_path}", flush=True)
        except BaseException as exc:
            record({"event": "failed", **identity, "error_type": type(exc).__name__, "error": str(exc)})
            raise
    result = {"schema": "train5_internal_batch_result_v1", "source_commit": plan["source_commit"],
              "protocol_sha256": plan["protocol_sha256"], "manifest_sha256": plan["manifest_sha256"],
              "runs": completed, "frozen_checkpoint_sha256": originals(),
              "scientific_admission": False, "heldout_evaluated": False}
    with (output / "batch-result.json").open("x", encoding="utf8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def _external_path(root, value):
    path = Path(value).resolve()
    if path == root or root in path.parents:
        raise ValueError("plan and signed packet must be outside the source checkout")
    return path


def main():
    from f1rl.research.wp7_protocol import _unique

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--evidence", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--valid-hours", type=int, default=168)
    sign = commands.add_parser("sign")
    sign.add_argument("--plan", required=True)
    sign.add_argument("--private-key", required=True)
    sign.add_argument("--packet", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--packet", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        destination = _external_path(root, args.plan)
        output = Path(args.output).resolve()
        if destination == output or output in destination.parents:
            raise ValueError("plan must be outside the future run directory")
        plan = build_plan(root, output, Path(args.evidence), valid_hours=args.valid_hours)
        _require_clean_source(root, plan["source_commit"])
        checks = preflight_checkpoints(root, args.evidence)
        if checks != FROZEN_CHECKPOINTS:
            raise ValueError("frozen checkpoint preflight differs")
        with destination.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
        print(f"Unsigned plan prepared: {destination}\nNo execution or approval has occurred.")
    elif args.command == "sign":
        from cryptography.hazmat.primitives.serialization import load_ssh_private_key

        destination = _external_path(root, args.packet)
        if destination.exists() or destination.is_symlink():
            raise ValueError("signed packet already exists")
        plan_path = _external_path(root, args.plan)
        plan = json.loads(plan_path.read_text(encoding="utf8"), object_pairs_hook=_unique)
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        print("\nSigning authorizes six internal simulation runs followed by 3600 TRAIN-only evaluation episodes, and acknowledges the listed missing evidence and pending review.")
        if input("Type DONG Y TRAIN5 NOI BO to sign, or anything else to cancel: ").strip() != "DONG Y TRAIN5 NOI BO":
            raise ValueError("owner did not confirm the displayed scope")
        password = getpass.getpass("Private-key passphrase (leave blank if unencrypted): ")
        try:
            key = load_ssh_private_key(Path(args.private_key).read_bytes(), password=password.encode() if password else None)
            signatures = sign_plan(root, plan, key)
        finally:
            password = None
        with destination.open("x", encoding="utf8", newline="\n") as handle:
            json.dump({"plan": plan, "envelopes": signatures}, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
        print(f"Signed requests saved: {destination}\nTraining has not started.")
    else:
        from scripts.wp7_train5_v4_development import evaluate_batch

        packet_path = _external_path(root, args.packet)
        packet = json.loads(packet_path.read_text(encoding="utf8"), object_pairs_hook=_unique)
        if not isinstance(packet, dict) or set(packet) != {"plan", "envelopes"}:
            raise ValueError("invalid signed batch packet")
        result = execute_batch(root, packet["plan"], packet["envelopes"])
        print("Six runs completed; starting TRAIN-only development evaluation.", flush=True)
        report = evaluate_batch(root, Path(packet["plan"]["output_root"]), Path(packet["plan"]["evidence_root"]))
        print(json.dumps({"runs": len(result["runs"]), "evaluation_episodes": report["summary"]["input_episode_count"],
                          "output_root": packet["plan"]["output_root"], "heldout_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
