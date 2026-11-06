"""One-click Windows TRAIN-only exploratory demo launcher.

The owner explicitly confirms the fresh scope before any Gym reset or PPO learning.
Never issues scientific authorization. Evidence lives outside the checkout.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[3]
# Always import this checkout, never a stale installed F1-RL from another worktree.
BOOTSTRAP = ROOT / "scripts" / "exploratory_ppo.py"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def state_summary(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return (data.get("status"), data.get("phase"), data.get("learning_timesteps"),
            data.get("optimizer_steps"))


def _runtime_ready(executable):
    try:
        check = subprocess.run(
            [str(executable), "-B", "-c",
             "import torch, gymnasium, stable_baselines3, yaml, cryptography"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        return check.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _interpreter_candidates():
    # Inspect only this F1-RL checkout and sibling worktrees, never arbitrary executables.
    if os.name != "nt":
        return []
    directories = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT.parent / ".venv" / "Scripts" / "python.exe",
    ]
    siblings = [ROOT.parent, ROOT.parent / ".worktrees" / "F1-RL"]
    for base in siblings:
        if base.is_dir():
            directories.extend(sorted(base.glob("*/.venv/Scripts/python.exe")))
    return directories


def ensure_runtime(command=None):
    if _runtime_ready(sys.executable):
        return None
    for interpreter in _interpreter_candidates():
        if str(interpreter).lower() != sys.executable.lower() and _runtime_ready(interpreter):
            print("[ENV] Using installed F1-RL Python:", interpreter, flush=True)
            return subprocess.call([str(interpreter), "-B", *(command or [str(Path(__file__).resolve())])], cwd=ROOT)
    raise ImportError("No Python interpreter with F1-RL RL/science dependencies was found")


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError("F1-RL requires Python 3.11+")
    alternate_result = ensure_runtime()
    if alternate_result is not None:
        return alternate_result
    from f1rl.exploratory.permit import LOCAL_ACKNOWLEDGEMENT, prepare

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    evidence = ROOT.parent / ".evidence" / "F1-RL" / "demo" / run_id
    output = evidence / "run"
    evidence.mkdir(parents=True, exist_ok=False)
    print("[PREFLIGHT] Checking clean source, runtime and TRAIN-only permit...", flush=True)
    permit = prepare(ROOT, output)
    permit_path = evidence / "permit.json"
    approval_path = evidence / "local-owner-consent.json"
    save_json(permit_path, permit)

    limits, seeds = permit["limits"], permit["seeds"]
    print("\nF1-RL | LOCAL EXPLORATORY PPO DEMO", flush=True)
    print("Source HEAD:", permit["source"].get("commit_sha", "recorded in permit"))
    print("Permit SHA:", permit["sha256"])
    print("Track:", permit["track"], "| profile:", permit["profile"])
    print("Schedule:", permit["execution"]["schedule"])
    print("Training:", limits["learning_timesteps"], "steps | checkpoint:", limits["checkpoint_steps"])
    print("CPU:", limits["threads"], "threads | GPU disabled | envs:", limits["n_envs"])
    print("Evaluation:", permit["evaluation"]["episodes_per_policy"], "episodes per policy")
    print("Training reward/curriculum:", permit.get('demo_training', {'version': 'registered'}))
    print("Timeout:", limits["wall_seconds"], "seconds | TRAIN-only; HELD-OUT sealed")
    print("Output:", output, flush=True)

    if input("\nType TRAIN to approve ONE local run (anything else cancels): ").strip() != "TRAIN":
        print("Cancelled. No training/rollout started.")
        return 0

    # Consent covers the exact fresh permit digest, one use and at most its wall budget.
    now = datetime.now(timezone.utc)
    issued = now - timedelta(seconds=1)
    expires = issued + timedelta(seconds=limits["wall_seconds"] - 2)
    envelope = {
        "approval_mode": "local_owner_consent_v1",
        "acknowledgement": LOCAL_ACKNOWLEDGEMENT,
        "payload": {
            "schema_version": "straight_exploratory_execution_v1",
            "purpose": "execute_exploratory_only",
            "permit_sha256": permit["sha256"],
            "issued_at_utc": issued.isoformat(),
            "expires_at_utc": expires.isoformat(),
        },
    }
    save_json(approval_path, envelope)

    command = [
        sys.executable, "-B", str(BOOTSTRAP), "execute",
        "--permit", str(permit_path), "--authorization", str(approval_path),
    ]
    print("\nTraining requested. State changes will appear below.\n", flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    last = None
    try:
        while process.poll() is None:
            status = state_summary(output / "state.json")
            if status is not None and status != last:
                print("[STATE]", "status=", status[0], "phase=", status[1],
                      "learning_steps=", status[2], "optimizer_steps=", status[3], flush=True)
                last = status
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nInterrupted locally. Inspect the supervisor and evidence before any new attempt.")
        # The supervisor/worker own cleanup and their compute boundaries.
        return 130

    status = state_summary(output / "state.json")
    if status is not None and status != last:
        print("[STATE]", status, flush=True)
    summary = output / "learning-summary.json"
    if summary.is_file():
        print("\nLEARNING SUMMARY:", summary, flush=True)
        print(summary.read_text(encoding="utf-8"), flush=True)
    print("\nFinal supervisor/evidence folder:", output, flush=True)
    print("Exit code:", process.returncode, flush=True)
    return int(process.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print("\n[LAUNCH FAILED]", f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("Check clean branch and Python extras: python -m pip install -e .[rl,science]", file=sys.stderr)
        raise SystemExit(2)
