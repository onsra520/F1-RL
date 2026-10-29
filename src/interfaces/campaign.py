"""Frozen, sequential WP3 LOCAL_DEMO training campaign orchestration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ._artifacts import (
    _atomic_json,
    _canonical,
    _strict_json,
    _write_new_json,
)
from ._campaign_reports import (
    _verify_evidence_manifest,
    _write_c0_gate_report,
    _write_evidence_manifest,
    _write_independent_review,
    _write_training_deliverables,
)
from ._campaign_validation import (
    _iter_jsonl,
    _validate_training_contact_terminal,
    validate_training_run,
)
from .campaign_contract import (
    CAMPAIGN_CONFIG,
    CAMPAIGN_EVENTS_SCHEMA,
    _expected_campaign_config,
    _load_campaign_config,
    build_campaign_manifest,
    model_schedule,
)

__all__ = ("CAMPAIGN_CONFIG", "CAMPAIGN_EVENTS_SCHEMA", "build_campaign_manifest",
           "model_schedule", "run_campaign", "validate_training_run",
           "_expected_campaign_config", "_iter_jsonl", "_load_campaign_config",
           "_validate_training_contact_terminal")


class _CampaignLock:
    """Windows byte-range lock; process exit releases it even after interruption."""
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None

    def __enter__(self):
        if os.name != "nt":
            raise OSError("WP3 campaign orchestration is local Windows only")
        import msvcrt
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError("another WP3 campaign process holds the run lock") from error
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()


def _append_campaign_event(path: Path, event: dict) -> None:
    payload = json.dumps({"schema_version": CAMPAIGN_EVENTS_SCHEMA, **event},
                         sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _state_update(path: Path, state: dict, **updates) -> dict:
    state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
    _atomic_json(path, state)
    return state


def _run_child(root: Path, output: Path, log_path: Path, model: dict,
               training: dict) -> tuple[int, float, dict]:
    command = [sys.executable, "-m", "scripts.training.local_simulation", "train",
               "--candidate", model["candidate_id"],
               "--tuning-seed-index", str(model["seed_index"]),
               "--output", str(output)]
    env = os.environ.copy()
    started = time.monotonic()
    peak_rss_bytes = 0
    peak_cpu_seconds = None
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            deadline = started + training["max_wall_seconds_per_model"] + 120
            while True:
                try:
                    returncode = proc.wait(timeout=min(10, max(0.1, deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    sample = _windows_process_sample(proc.pid)
                    if sample is not None:
                        peak_rss_bytes = max(peak_rss_bytes, sample["rss_bytes"])
                        peak_cpu_seconds = sample["cpu_seconds"]
                    if time.monotonic() >= deadline:
                        if os.name == "nt":
                            proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                        try:
                            proc.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        raise TimeoutError("campaign parent wall-time guard expired; child evidence preserved")
        except KeyboardInterrupt:
            proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    return returncode, time.monotonic() - started, {
        "peak_working_set_bytes": peak_rss_bytes or None,
        "final_cpu_seconds": peak_cpu_seconds,
        "sampling_interval_seconds": 10,
        "measurement_source": "Windows GetProcessMemoryInfo/GetProcessTimes" if os.name == "nt" else "unavailable",
    }


def _windows_process_sample(pid: int) -> dict | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    handle_type = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = handle_type
    kernel.GetProcessTimes.argtypes = [handle_type, ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
                                       ctypes.POINTER(FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [handle_type]
    kernel.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [handle_type, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x410, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        memory_ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        creation, exit_time, kernel_time, user_time = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        times_ok = kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time))
        cpu_seconds = None
        if times_ok:
            ticks = ((kernel_time.high << 32) | kernel_time.low) + ((user_time.high << 32) | user_time.low)
            cpu_seconds = ticks / 10_000_000
        return {"rss_bytes": int(counters.PeakWorkingSetSize) if memory_ok else 0,
                "cpu_seconds": cpu_seconds}
    finally:
        kernel.CloseHandle(handle)


def _load_campaign(root: Path, output: Path) -> tuple[dict, dict, list[dict], dict, Path]:
    expected_manifest, config, schedule = build_campaign_manifest(root)
    manifest, _ = _strict_json(output / "WP3-CAMPAIGN-MANIFEST.json")
    state, _ = _strict_json(output / "campaign-state.json")
    for key in ("source_commit", "source_tree_sha256", "baseline_commit", "campaign_config_sha256",
                "simulation_only_config_sha256", "registration_sha256", "track_manifest_sha256",
                "training_profile_sha256", "protocol_sha256"):
        if manifest.get(key) != expected_manifest.get(key):
            raise ValueError(f"campaign resume identity drifted at {key}")
    if manifest.get("sha256") != sha256(_canonical({k: v for k, v in manifest.items() if k != "sha256"})).hexdigest():
        raise ValueError("campaign manifest self-checksum mismatch")
    if state.get("campaign_manifest_sha256") != manifest["sha256"]:
        raise ValueError("campaign state is not bound to its immutable manifest")
    return manifest, config, schedule, state, output


def _validate_or_reuse_run(root: Path, output: Path, expected: dict,
                           manifest: dict, config: dict) -> dict | None:
    if not output.exists():
        return None
    if not output.is_dir() or output.is_symlink():
        raise ValueError("model output path exists but is not a regular directory")
    try:
        validation = validate_training_run(root, output, expected, manifest, config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"existing model output is incomplete or invalid; refusing duplicate training: {output}") from error
    validation_path = output / "engineering-validation.json"
    if validation_path.exists():
        prior, _ = _strict_json(validation_path)
        calculated = sha256(_canonical({k: v for k, v in prior.items() if k != "sha256"})).hexdigest()
        expected_hash = sha256(_canonical(validation)).hexdigest()
        if prior.get("sha256") != calculated or prior.get("sha256") != expected_hash:
            raise ValueError("existing engineering validation checksum mismatch")
    else:
        validation["sha256"] = sha256(_canonical(validation)).hexdigest()
        _write_new_json(validation_path, validation)
    return validation


def _new_campaign(root: Path, output: Path) -> tuple[dict, dict, list[dict], dict, Path]:
    if output.exists():
        raise FileExistsError(f"campaign evidence root already exists; use resume after review: {output}")
    manifest, config, schedule = build_campaign_manifest(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "logs").mkdir()
    (output / "evaluation").mkdir()
    from ._evaluation_roster import build_evaluation_roster
    roster = build_evaluation_roster(root, manifest, config)
    _write_new_json(output / "WP3-CAMPAIGN-MANIFEST.json", manifest)
    _write_new_json(output / "evaluation" / "evaluation-roster.json", roster)
    state = {
        "schema_version": "f1rl_wp3_campaign_state_v1",
        "campaign_manifest_sha256": manifest["sha256"],
        "status": "PREPARED", "phase": "C0_ENGINEERING_GATE",
        "current_model_id": None, "completed_model_ids": [],
        "models": {row["model_id"]: {"status": "pending"} for row in schedule},
        "completed_training_timesteps": 0, "updated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_evidence": False, "heldout_access": False,
        "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    _write_new_json(output / "campaign-state.json", state)
    return manifest, config, schedule, state, output


def run_campaign(repository_root: str | Path, output_directory: str | Path,
                 *, resume: bool = False) -> dict:
    root = Path(repository_root).resolve(strict=True)
    output_arg = Path(output_directory)
    if not output_arg.is_absolute():
        raise ValueError("WP3 campaign evidence root must be an absolute path")
    output = output_arg.resolve()
    if output.is_relative_to(root):
        raise ValueError("campaign evidence must be outside the repository")
    if not resume:
        manifest, config, schedule, state, output = _new_campaign(root, output)
    else:
        manifest, config, schedule, state, output = _load_campaign(root, output)
    lock_path = output / ".campaign.lock"
    events_path = output / "campaign-events.jsonl"
    try:
        with _CampaignLock(lock_path):
            if (resume and state.get("status") in ("COMPLETE", "EVALUATION_WITH_INVALID_CELLS")
                    and (output / "WP3-EVIDENCE-MANIFEST.json").exists()):
                _verify_evidence_manifest(output)
                return state
            verified_in_process: set[str] = set()
            if resume:
                for expected in schedule:
                    model_id = expected["model_id"]
                    if state["models"][model_id].get("status") == "ENGINEERING_PASS":
                        run_dir = output / expected["run_relative_path"]
                        validation = _validate_or_reuse_run(root, run_dir, expected, manifest, config)
                        if validation is None:
                            raise ValueError(f"completed campaign state has no valid model evidence: {model_id}")
                        verified_in_process.add(model_id)
            # A completed, valid run is re-used; an incomplete run is never retrained implicitly.
            for expected in schedule:
                model_id = expected["model_id"]
                if model_id in verified_in_process:
                    continue
                run_dir = output / expected["run_relative_path"]
                valid = None
                if run_dir.exists():
                    valid = _validate_or_reuse_run(root, run_dir, expected, manifest, config)
                if valid is not None:
                    entry = state["models"][model_id]
                    entry.update(status="ENGINEERING_PASS", validation=valid,
                                 output_path=run_dir.relative_to(output).as_posix())
                    if model_id not in state["completed_model_ids"]:
                        state["completed_model_ids"].append(model_id)
                    state["completed_training_timesteps"] = len(state["completed_model_ids"]) * config["training"]["timesteps_per_model"]
                    verified_in_process.add(model_id)
                    continue
                if state["models"][model_id]["status"] not in ("pending", "launching"):
                    raise RuntimeError(f"model {model_id} is not retryable; refusing duplicate training")
                if state.get("current_model_id") not in (None, model_id):
                    raise RuntimeError("campaign state points to a different unfinished model")
                state = _state_update(output / "campaign-state.json", state,
                                      status="RUNNING", phase=("C0_ENGINEERING_GATE" if model_id == "C-seed0" else "STAGE_2_TRAINING"),
                                      current_model_id=model_id)
                state["models"][model_id].update(status="launching", output_path=expected["run_relative_path"])
                _atomic_json(output / "campaign-state.json", state)
                log_path = output / "logs" / f"{model_id}.log"
                _append_campaign_event(events_path, {"event": "model_started", "model_id": model_id,
                                                      "candidate_id": expected["candidate_id"],
                                                      "seed_index": expected["seed_index"],
                                                      "source_commit": manifest["source_commit"],
                                                      "timestamp": datetime.now(timezone.utc).isoformat()})
                state["models"][model_id].update(status="running", started_at=datetime.now(timezone.utc).isoformat())
                _atomic_json(output / "campaign-state.json", state)
                try:
                    returncode, elapsed, resource_usage = _run_child(root, run_dir, log_path, expected, config["training"])
                except KeyboardInterrupt:
                    state["models"][model_id].update(status="interrupted", log_path=log_path.relative_to(output).as_posix())
                    _state_update(output / "campaign-state.json", state, status="INTERRUPTED",
                                  phase="INTERRUPTED_PRESERVED", current_model_id=model_id)
                    _append_campaign_event(events_path, {"event": "model_interrupted", "model_id": model_id,
                                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                    raise
                except BaseException as error:
                    state["models"][model_id].update(status="failed", error=f"{type(error).__name__}: {error}")
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    _append_campaign_event(events_path, {"event": "model_failed", "model_id": model_id,
                                                          "error": f"{type(error).__name__}: {error}",
                                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                    raise
                state["models"][model_id].update(returncode=returncode, elapsed_seconds=elapsed,
                                                 resource_usage=resource_usage,
                                                 log_path=log_path.relative_to(output).as_posix())
                if returncode != 0:
                    state["models"][model_id]["status"] = "failed"
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    raise RuntimeError(f"LOCAL_DEMO training child returned {returncode} for {model_id}")
                try:
                    validation = validate_training_run(root, run_dir, expected, manifest, config)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    state["models"][model_id].update(status="integrity_failed", error=f"{type(error).__name__}: {error}")
                    _state_update(output / "campaign-state.json", state, status="ENGINEERING_FAILED",
                                  phase="STOPPED_ON_FAILURE", current_model_id=model_id)
                    raise RuntimeError(f"engineering validation failed for {model_id}") from error
                validation["sha256"] = sha256(_canonical(validation)).hexdigest()
                _write_new_json(run_dir / "engineering-validation.json", validation)
                if model_id == "C-seed0":
                    lifecycle = validation["lifecycle"]
                    uncovered = [track for track, count in lifecycle["track_timestep_counts"].items() if count <= 0]
                    gate_pass = not uncovered and lifecycle["automatic_resets_verified"] >= 1
                    if not gate_pass:
                        reason = (f"unobserved TRAIN tracks={uncovered}; verified automatic reset boundaries="
                                  f"{lifecycle['automatic_resets_verified']}")
                        _write_c0_gate_report(output / "WP3-C0-ENGINEERING-GATE.md",
                                              validation, "ENGINEERING_FAIL", reason)
                        state["models"][model_id].update(status="integrity_failed", engineering_failure={
                            "unobserved_tracks": uncovered,
                            "automatic_resets_verified": lifecycle["automatic_resets_verified"],
                        })
                        _state_update(output / "campaign-state.json", state,
                                      status="ENGINEERING_FAILED", phase="STOPPED_ON_FAILURE",
                                      current_model_id=model_id)
                        raise RuntimeError("C0 failed actual multi-track coverage/autoreset gates")
                state["models"][model_id].update(status="ENGINEERING_PASS", validation=validation)
                if model_id not in state["completed_model_ids"]:
                    state["completed_model_ids"].append(model_id)
                state["completed_training_timesteps"] = len(state["completed_model_ids"]) * config["training"]["timesteps_per_model"]
                if model_id == "C-seed0":
                    _write_c0_gate_report(output / "WP3-C0-ENGINEERING-GATE.md",
                                          validation, "ENGINEERING_PASS")
                    state["c0_engineering_gate"] = {
                        "status": "ENGINEERING_PASS", "model_id": model_id,
                        "validation_sha256": validation["sha256"],
                        "passed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    state["phase"] = "STAGE_2_TRAINING"
                state["current_model_id"] = None
                verified_in_process.add(model_id)
                _atomic_json(output / "campaign-state.json", state)
                _append_campaign_event(events_path, {"event": "model_engineering_pass", "model_id": model_id,
                                                      "learning_timesteps": validation["learning_timesteps"],
                                                      "optimizer_steps": validation["optimizer_steps"],
                                                      "telemetry_sha256": validation["telemetry"]["sha256"],
                                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            if len(state["completed_model_ids"]) != 12:
                raise RuntimeError("campaign training phase did not validate all 12 models")
            _write_training_deliverables(output, state, schedule, manifest, config)
            state = _state_update(output / "campaign-state.json", state,
                                  status="TRAINING_COMPLETE", phase="STAGE_3_TRAIN_EVALUATION",
                                  current_model_id=None)
            from .evaluation import run_evaluation
            evaluation = run_evaluation(root, output, manifest, config, schedule)
            state = _state_update(output / "campaign-state.json", state,
                                  status=("COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_WITH_INVALID_CELLS"),
                                  phase="COMPLETE" if evaluation["status"] == "PASS" else "EVALUATION_REVIEW",
                                  evaluation=evaluation)
            _write_independent_review(output, schedule, state, evaluation)
            _write_evidence_manifest(output)
            return state
    except KeyboardInterrupt:
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Sequential frozen WP3 local TRAIN PPO tuning campaign")
    parser.add_argument("command", choices=("run", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    state = run_campaign(root, args.output, resume=args.command == "resume")
    print(f"[WP3] {state['status']} completed={len(state['completed_model_ids'])}/12 "
          f"timesteps={state['completed_training_timesteps']}", flush=True)
    return 0 if state["status"] == "COMPLETE" else 2
