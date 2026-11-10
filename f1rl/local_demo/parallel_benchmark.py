from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .campaign import _atomic_json, _canonical, _run_bounded_stage, _write_new_json
from .process_resources import (WindowsKillJob, _process_rows, sample_process_tree,
                                windows_cpu_load_percent)


_MOCK_WORKER = """
import json
from pathlib import Path
import sys
import time

duration = float(sys.argv[1])
memory_mib = int(sys.argv[2])
ready_path = Path(sys.argv[3])
result_path = Path(sys.argv[4])
payload = bytearray(memory_mib * 1024 * 1024)
for index in range(0, len(payload), 4096):
    payload[index] = index % 251
ready_path.write_text('ready', encoding='utf-8')
started = time.monotonic()
time.sleep(duration)
result_path.write_text(json.dumps({'mock_operations': len(payload) // 4096}), encoding='utf-8')
"""


def _measure_mock_worker(job: dict, *, abort_event: threading.Event) -> dict:
    directory = Path(job["directory"])
    log_path = directory / f"{job['model_id']}.log"
    ready_path = directory / f"{job['model_id']}.ready"
    result_path = directory / f"{job['model_id']}.result.json"
    identity_path = directory / f"{job['model_id']}.process.json"
    started = time.monotonic()
    peak_rss = 0
    cpu_seconds = 0.0
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", _MOCK_WORKER,
             str(job["duration_seconds"]), str(job["memory_mib"]),
             str(ready_path), str(result_path)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        process_job = WindowsKillJob()
        cleanup_started = time.monotonic()
        try:
            process_job.assign(proc.pid)
            first_sample_at = None
            while proc.poll() is None:
                sample = sample_process_tree(proc.pid)
                if sample is not None:
                    peak_rss = max(peak_rss, sample["rss_bytes"])
                    cpu_seconds = max(cpu_seconds, sample["cpu_seconds"])
                    root_row = next((row for row in sample["processes"] if row["pid"] == proc.pid), None)
                    if root_row and not identity_path.exists():
                        _atomic_json(identity_path, {
                            "pid": proc.pid,
                            "creation_time_100ns": root_row["creation_time_100ns"],
                            "model_id": job["model_id"],
                        })
                if ready_path.exists() and first_sample_at is None:
                    first_sample_at = time.monotonic()
                if abort_event.is_set():
                    try:
                        proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                    except (OSError, ProcessLookupError):
                        pass
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process_job.terminate()
                        proc.wait(timeout=5)
                    if not process_job.wait_empty(5):
                        raise TimeoutError("mock benchmark subprocess cleanup did not finish")
                    return {"returncode": proc.returncode, "cancelled": True,
                            "cleanup_seconds": time.monotonic() - cleanup_started}
                time.sleep(0.025)
            if proc.returncode != 0:
                return {"returncode": proc.returncode, "cancelled": False,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_rss_bytes": peak_rss, "cpu_seconds": cpu_seconds}
            process_job.terminate()
            if not process_job.wait_empty(5):
                raise TimeoutError("mock benchmark subprocess cleanup did not finish")
            process_job.close()
            cleanup_seconds = time.monotonic() - cleanup_started
            remaining = {row["pid"] for row in _process_rows()}
            result = json.loads(result_path.read_text(encoding="utf-8"))
            return {
                "returncode": 0, "cancelled": False,
                "elapsed_seconds": time.monotonic() - started,
                "startup_seconds": ((first_sample_at - started) if first_sample_at is not None else None),
                "cleanup_seconds": cleanup_seconds,
                "peak_rss_bytes": peak_rss or None,
                "cpu_seconds": cpu_seconds,
                "mock_operations": result["mock_operations"],
                "process_cleanup_complete": proc.pid not in remaining,
                "pid": proc.pid,
            }
        finally:
            if proc.poll() is None:
                process_job.terminate()
                proc.wait(timeout=5)
            process_job.close()


def run_mock_subprocess_benchmark(output_directory: str | Path, *,
                                  job_durations_seconds: list[float] | None = None,
                                  memory_mib_per_job: int = 8) -> dict:
    if os.name != "nt":
        raise OSError("mock scheduler process benchmark is supported on Windows only")
    output_arg = Path(output_directory)
    if not output_arg.is_absolute():
        raise ValueError("benchmark output directory must be absolute")
    output = output_arg.resolve()
    if output.exists():
        raise FileExistsError("benchmark output directory already exists")
    durations = job_durations_seconds or [0.10, 0.28, 0.16, 0.40, 0.12, 0.32, 0.20, 0.24]
    if (len(durations) < 4 or any(type(value) not in (int, float) or value <= 0 for value in durations)
            or any(value > 60 for value in durations)
            or type(memory_mib_per_job) is not int or not 1 <= memory_mib_per_job <= 64):
        raise ValueError("mock workload requires four positive durations and 1-256 MiB per child")
    output.mkdir(parents=True, exist_ok=False)
    cpu_before = windows_cpu_load_percent()
    measurements = []
    benchmark_jobs = [
        {"model_id": f"mock-{index:03d}", "duration_seconds": float(duration),
         "memory_mib": memory_mib_per_job}
        for index, duration in enumerate(durations)
    ]
    for workers in (1, 2, 4):
        run_dir = output / f"workers-{workers}"
        run_dir.mkdir()
        jobs = [{**job, "directory": str(run_dir)} for job in benchmark_jobs]
        runtime = {"active_worker_seconds": 0.0, "worker_capacity_seconds": 0.0,
                   "peak_aggregate_rss_bytes": 0, "max_process_count": 0}
        def on_tick(active_jobs: tuple[dict, ...], delta_seconds: float) -> None:
            runtime["active_worker_seconds"] += len(active_jobs) * delta_seconds
            runtime["worker_capacity_seconds"] += workers * delta_seconds
            rss_total = 0
            process_count = 0
            for job in active_jobs:
                identity_path = run_dir / f"{job['model_id']}.process.json"
                if not identity_path.is_file():
                    continue
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
                sample = sample_process_tree(identity["pid"])
                if sample is None:
                    continue
                root_row = next((row for row in sample["processes"]
                                 if row["pid"] == identity["pid"]), None)
                if root_row is None or root_row["creation_time_100ns"] != identity["creation_time_100ns"]:
                    continue
                rss_total += sample["rss_bytes"]
                process_count += sample["process_count"]
            runtime["peak_aggregate_rss_bytes"] = max(runtime["peak_aggregate_rss_bytes"], rss_total)
            runtime["max_process_count"] = max(runtime["max_process_count"], process_count + 1)

        stage_started = time.monotonic()
        stage = _run_bounded_stage(
            jobs, workers, _measure_mock_worker,
            lambda _job, _result: {"status": "ENGINEERING_PASS"},
            on_tick=on_tick,
            tick_interval_seconds=0.05,
        )
        wall_seconds = time.monotonic() - stage_started
        if stage["status"] != "ENGINEERING_PASS":
            raise RuntimeError(f"mock subprocess benchmark failed with {stage['failures']}")
        rows = []
        for job in benchmark_jobs:
            outcome = stage["outcomes"][job["model_id"]]
            rows.append({
                "model_id": job["model_id"],
                "planned_duration_seconds": job["duration_seconds"],
                "elapsed_seconds": outcome.get("elapsed_seconds"),
                "startup_seconds": outcome.get("startup_seconds"),
                "cleanup_seconds": outcome.get("cleanup_seconds"),
                "peak_rss_bytes": outcome.get("peak_rss_bytes"),
                "cpu_seconds": outcome.get("cpu_seconds"),
                "mock_operations": outcome.get("mock_operations"),
                "pid": outcome.get("pid"),
                "process_cleanup_complete": outcome.get("process_cleanup_complete", False),
                "status": "MOCK_SUBPROCESS_PASS",
            })
        raw_path = run_dir / "raw-measurements.jsonl"
        with raw_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        total_cpu_seconds = sum(row["cpu_seconds"] or 0.0 for row in rows)
        run_result = {
            "worker_count": workers, "backend": "bounded_thread_supervisor_subprocess",
            "campaign_wall_seconds": wall_seconds,
            "completed_jobs": len(rows),
            "completed_jobs_per_hour": len(rows) * 3600 / max(wall_seconds, 0.001),
            "speedup_vs_sequential": None,
            "worker_utilization": (runtime["active_worker_seconds"] / runtime["worker_capacity_seconds"]
                                   if runtime["worker_capacity_seconds"] else 0.0),
            "scheduler_idle_seconds": max(0.0, runtime["worker_capacity_seconds"]
                                          - runtime["active_worker_seconds"]),
            "aggregate_cpu_percent": (100 * total_cpu_seconds /
                                      max(wall_seconds * (os.cpu_count() or 1), 0.001)),
            "aggregate_peak_rss_bytes": runtime["peak_aggregate_rss_bytes"],
            "peak_process_count_including_supervisor": runtime["max_process_count"],
            "per_job": rows,
            "training_transitions_per_second": None,
            "ppo_rollout_seconds": None, "ppo_optimizer_seconds": None,
            "failure_count": 0, "retry_count": 0,
            "process_cleanup_complete": all(row["process_cleanup_complete"] for row in rows),
            "raw_measurements_path": raw_path.name,
            "campaign_completion": False,
            "scientific_evidence": False, "heldout_access": False,
        }
        measurements.append(run_result)
    sequential_wall = measurements[0]["campaign_wall_seconds"]
    for row in measurements:
        row["speedup_vs_sequential"] = sequential_wall / max(row["campaign_wall_seconds"], 0.001)
    cpu_after = windows_cpu_load_percent()
    repository_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root,
                                   check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repository_root,
                                check=True, capture_output=True, text=True).stdout.strip())
    source_digest = sha256()
    for relative in ("f1rl/local_demo/campaign.py", "f1rl/local_demo/process_resources.py",
                     "f1rl/local_demo/parallel_benchmark.py"):
        source_digest.update(relative.encode("utf-8"))
        source_digest.update((repository_root / relative).read_bytes())
    report = {
        "schema_version": "f1rl_wp3_parallel_scheduler_benchmark_v1",
        "benchmark_kind": "mock_subprocess_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": {"logical_processors": os.cpu_count() or 1,
                 "cpu_load_before_percent": cpu_before,
                 "cpu_load_after_percent": cpu_after,
                 "cpu_load_measurement": "Windows GetSystemTimes"},
        "source": {"commit": source_commit, "worktree_dirty": dirty,
                   "scheduler_source_sha256": source_digest.hexdigest()},
        "workload": {"job_durations_seconds": durations,
                     "memory_mib_per_job": memory_mib_per_job,
                     "job_ids": [job["model_id"] for job in benchmark_jobs],
                     "seed_design": "fixed job-index order; no random sampling",
                     "command_template": ["python", "-c", "embedded mock worker",
                                          "duration_seconds", "memory_mib", "ready_path", "result_path"]},
        "comparisons": measurements,
        "campaign_completion": False, "scientific_evidence": False,
        "heldout_access": False, "wp7_status": "BLOCKED", "gate2_status": "BLOCKED",
    }
    report["sha256"] = sha256(_canonical(report)).hexdigest()
    _write_new_json(output / "MOCK-SCHEDULER-BENCHMARK.json", report)
    return report
