from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import f1rl.local_demo.evaluation as evaluation
import f1rl.local_demo.parallel_campaign as campaign
import f1rl.local_demo.process_resources as process_resources
from f1rl.local_demo.parallel_campaign import (
    PARALLEL_CAMPAIGN_CONFIG,
    _expected_parallel_campaign_config,
    load_parallel_contract,
    run_parallel_schedule,
    validate_parallel_contract,
    validate_parallel_resources,
)
from f1rl.local_demo.process_resources import WindowsKillJob, _process_rows, process_tree_pids
from f1rl.local_demo.parallel_benchmark import run_mock_subprocess_benchmark
from f1rl.local_demo.admission import _source_identity


ROOT = Path(__file__).parents[2]
GIB = 1024 ** 3
BASE_CAMPAIGN_SHA256 = "8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4"
TRAIN_TRACKS = ("straight-sprint-v1", "slalom-v1", "right-turn-v1", "ramp-v1")


def _write_chart_run(root: Path, relative: str, loss: float, *, telemetry_override: bytes | None = None) -> Path:
    run_dir = root / relative
    run_dir.mkdir(parents=True)
    payload = telemetry_override or (
        json.dumps({"event_type": "ppo_update", "metrics": {"train/loss": loss}}) + "\n"
    ).encode("utf-8")
    telemetry = run_dir / "training-telemetry.jsonl"
    telemetry.write_bytes(payload)
    (run_dir / "engineering-validation.json").write_text(json.dumps({
        "status": "ENGINEERING_PASS", "model_id": "A-seed0",
        "telemetry": {"path": telemetry.name, "bytes": len(payload), "sha256": sha256(payload).hexdigest()},
    }), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("run_relative_path", ["runs/A-seed0", "runs/A-seed0/attempt-0001"])
def test_learning_chart_reads_sequential_and_parallel_run_paths(tmp_path, run_relative_path):
    _write_chart_run(tmp_path, run_relative_path, 2.5)
    model = {"model_id": "A-seed0", "candidate_id": "A",
             "run_relative_path": run_relative_path}

    evaluation._write_learning_chart(tmp_path, [model], {"models": [model]})

    chart = (tmp_path / "WP3-LEARNING-PROGRESS.svg").read_text(encoding="utf-8")
    assert ">2.5<" in chart


def test_learning_chart_uses_only_the_validated_retry_attempt(tmp_path):
    _write_chart_run(tmp_path, "runs/A-seed0/attempt-0001", 99.0)
    _write_chart_run(tmp_path, "runs/A-seed0/attempt-0002", 2.5)
    model = {"model_id": "A-seed0", "candidate_id": "A",
             "run_relative_path": "runs/A-seed0/attempt-0002"}

    evaluation._write_learning_chart(tmp_path, [model], {"models": [model]})

    chart = (tmp_path / "WP3-LEARNING-PROGRESS.svg").read_text(encoding="utf-8")
    assert ">2.5<" in chart
    assert ">99<" not in chart


@pytest.mark.parametrize("telemetry_override", [None, b'{"event_type":"ppo_update"}\n'])
def test_learning_chart_fails_closed_on_missing_or_mismatched_telemetry(tmp_path, telemetry_override):
    relative = "runs/A-seed0/attempt-0001"
    run_dir = _write_chart_run(tmp_path, relative, 2.5)
    telemetry = run_dir / "training-telemetry.jsonl"
    if telemetry_override is None:
        telemetry.unlink()
        expected = FileNotFoundError
    else:
        telemetry.write_bytes(telemetry_override)
        expected = ValueError
    model = {"model_id": "A-seed0", "candidate_id": "A", "run_relative_path": relative}

    with pytest.raises(expected):
        evaluation._write_learning_chart(tmp_path, [model], {"models": [model]})


def test_parallel_contract_pins_v3_without_changing_its_bytes():
    config, contract_sha256 = load_parallel_contract(ROOT)
    v3_path = ROOT / "configs/research/wp3-ppo-tuning-campaign-v3.json"
    parallel_path = ROOT / PARALLEL_CAMPAIGN_CONFIG

    assert sha256(v3_path.read_bytes()).hexdigest() == BASE_CAMPAIGN_SHA256
    assert config == _expected_parallel_campaign_config()
    assert config["base_campaign_config_sha256"] == BASE_CAMPAIGN_SHA256
    assert contract_sha256 == sha256(parallel_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", [
    {"allowed_concurrency": [1, 3]},
    {"scheduler": {"gate_model_id": "A-seed0"}},
    {"resources": {"per_worker_torch_threads": 4}},
    {"scientific_training_authorized": True},
    {"heldout_access": True},
])
def test_parallel_contract_rejects_policy_drift(mutation):
    config = _expected_parallel_campaign_config()
    config.update(mutation)

    with pytest.raises(ValueError):
        validate_parallel_contract(config, BASE_CAMPAIGN_SHA256)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_resource_budget_bounds_processes_threads_and_rss(workers):
    budget = validate_parallel_resources(
        _expected_parallel_campaign_config(),
        workers,
        logical_processors=32,
        available_memory_bytes=26 * GIB,
    )

    assert budget["training_processes"] == workers
    assert budget["total_processes"] == workers + 1
    assert budget["torch_intraop_threads"] == workers * 8
    assert budget["torch_interop_threads"] == workers
    assert budget["process_tree_rss_limit_bytes"] == workers * 4 * GIB


def test_parallel_resource_budget_refuses_insufficient_host_capacity():
    config = _expected_parallel_campaign_config()

    with pytest.raises(ValueError, match="logical processor"):
        validate_parallel_resources(config, 4, logical_processors=31,
                                    available_memory_bytes=26 * GIB)
    with pytest.raises(ValueError, match="available memory"):
        validate_parallel_resources(config, 4, logical_processors=32,
                                    available_memory_bytes=20 * GIB)


def _schedule(*model_ids: str) -> list[dict]:
    return [{"model_id": model_id, "candidate_id": model_id.split("-seed")[0],
             "seed_index": int(model_id.rsplit("seed", 1)[1])}
            for model_id in model_ids]


def _valid_training_validation(job: dict, _result: dict | None = None) -> dict:
    validation = {"status": "ENGINEERING_PASS", "model_id": job["model_id"],
                  "learning_timesteps": 65536}
    if job["model_id"] == "C-seed0":
        validation["lifecycle"] = {
            "track_timestep_counts": {track: 1 for track in TRAIN_TRACKS},
            "automatic_resets_verified": 1,
        }
    return validation


def test_c0_gate_failure_does_not_start_follow_up_jobs():
    started = []
    schedule = _schedule("C-seed0", "A-seed0", "B-seed0", "D-seed0")

    def run_child(job, *, abort_event):
        started.append(job["model_id"])
        return {"returncode": 0, "elapsed_seconds": 0.01}

    def validate_job(job, _result):
        validation = _valid_training_validation(job)
        if job["model_id"] == "C-seed0":
            validation["lifecycle"]["automatic_resets_verified"] = 0
        return validation

    outcome = run_parallel_schedule(schedule, 4, run_child, validate_job)

    assert outcome["status"] == "C0_ENGINEERING_FAILED"
    assert started == ["C-seed0"]
    assert outcome["pending_model_ids"] == ["A-seed0", "B-seed0", "D-seed0"]


def test_c0_short_budget_cannot_open_parallel_queue():
    started = []
    schedule = _schedule("C-seed0", "A-seed0")

    def run_child(job, *, abort_event):
        started.append(job["model_id"])
        return {"returncode": 0, "elapsed_seconds": 0.01}

    def validate_job(job, _result):
        validation = _valid_training_validation(job)
        if job["model_id"] == "C-seed0":
            validation["learning_timesteps"] = 4096
        return validation

    outcome = run_parallel_schedule(schedule, 2, run_child, validate_job)

    assert outcome["status"] == "C0_ENGINEERING_FAILED"
    assert started == ["C-seed0"]


def test_dynamic_scheduler_fills_freed_worker_before_slowest_job_finishes():
    schedule = _schedule("C-seed0", "A-seed0", "B-seed0", "D-seed0")
    lock = threading.Lock()
    active = 0
    peak_active = 0
    started_at = {}
    finished_at = {}

    def run_child(job, *, abort_event):
        nonlocal active, peak_active
        model_id = job["model_id"]
        duration = {"C-seed0": 0.01, "A-seed0": 0.35,
                    "B-seed0": 0.04, "D-seed0": 0.04}[model_id]
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            started_at[model_id] = time.monotonic()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if abort_event.is_set():
                return {"returncode": -1, "cancelled": True}
            time.sleep(0.005)
        with lock:
            finished_at[model_id] = time.monotonic()
            active -= 1
        return {"returncode": 0, "elapsed_seconds": duration}

    outcome = run_parallel_schedule(schedule, 2, run_child, _valid_training_validation)

    assert outcome["status"] == "ENGINEERING_PASS"
    assert peak_active <= 2
    assert started_at["D-seed0"] < finished_at["A-seed0"]
    assert set(outcome["completed_model_ids"]) == {job["model_id"] for job in schedule}


def test_failed_worker_stops_queue_and_cancels_active_sibling():
    schedule = _schedule("C-seed0", "A-seed0", "B-seed0", "D-seed0")
    started = []
    cancelled = threading.Event()

    def run_child(job, *, abort_event):
        model_id = job["model_id"]
        started.append(model_id)
        if model_id == "A-seed0":
            return {"returncode": 5, "elapsed_seconds": 0.01}
        if model_id == "B-seed0":
            while not abort_event.wait(0.005):
                pass
            cancelled.set()
            return {"returncode": -1, "cancelled": True}
        return {"returncode": 0, "elapsed_seconds": 0.01}

    outcome = run_parallel_schedule(schedule, 2, run_child, _valid_training_validation)

    assert outcome["status"] == "ENGINEERING_FAILED"
    assert outcome["failed_model_id"] == "A-seed0"
    assert "D-seed0" not in started
    assert cancelled.is_set()


def _full_test_schedule():
    rows = []
    for order, (candidate, seed_index) in enumerate([
            ("C", 0), ("A", 0), ("B", 0), ("D", 0),
            ("A", 1), ("B", 1), ("C", 1), ("D", 1),
            ("A", 2), ("B", 2), ("C", 2), ("D", 2)]):
        rows.append({
            "order": order, "model_id": f"{candidate}-seed{seed_index}",
            "candidate_id": candidate, "seed_index": seed_index,
            "seed_bundle": {"ppo": 100 + seed_index, "sampling": 200 + seed_index,
                            "training_reset": 300 + seed_index},
            "candidate_factors": {"candidate_id": candidate, "training_corridor_m": 2.0},
            "run_relative_path": f"runs/{candidate}-seed{seed_index}",
        })
    return rows


def test_campaign_level_c0_failure_writes_no_queue_decision(tmp_path, monkeypatch):
    config, _raw, config_sha = campaign._load_campaign_config(ROOT)
    schedule = _full_test_schedule()
    base_manifest = {
        "schema_version": "f1rl_wp3_campaign_manifest_v2",
        "sha256": "base-manifest-test-sha256", "source_commit": "test-source-commit",
        "source_tree_sha256": "test-source-tree-sha256",
        "campaign_config_sha256": config_sha, "training": config["training"],
    }
    parallel_config = _expected_parallel_campaign_config()
    parallel_sha = sha256((ROOT / PARALLEL_CAMPAIGN_CONFIG).read_bytes()).hexdigest()
    monkeypatch.setattr(campaign, "build_campaign_manifest",
                        lambda _root: (base_manifest, config, schedule))
    monkeypatch.setattr(campaign, "load_parallel_contract",
                        lambda _root: (parallel_config, parallel_sha))
    monkeypatch.setattr(campaign, "_parallel_preflight",
                        lambda *_args: {"logical_processors": 32, "available_memory_bytes": 40 * GIB,
                                        "host_cpu_load_percent": 2.0})
    monkeypatch.setattr(campaign, "validate_training_run", lambda _root, _run, job, *_args: {
        "schema_version": "f1rl_wp3_model_engineering_validation_v1",
        "status": "ENGINEERING_PASS", "model_id": job["model_id"],
        "learning_timesteps": 65536, "lifecycle": {
            "track_timestep_counts": {track: 1 for track in TRAIN_TRACKS},
            "automatic_resets_verified": 0,
        },
    })
    started = []

    def fake_child(_root, run_path, log_path, job, _training, **_kwargs):
        started.append(job["model_id"])
        run_path.mkdir(parents=True, exist_ok=False)
        log_path.write_text("simulated child\n", encoding="utf-8")
        return {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.01,
                "cleanup_seconds": 0.0, "resource_usage": {}}

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    import f1rl.local_demo.evaluation as evaluation
    monkeypatch.setattr(evaluation, "build_evaluation_roster", lambda *_args: {"sha256": "test-roster"})
    output = tmp_path / "parallel-campaign"

    state = campaign.run_parallel_campaign(ROOT, output, workers=4)

    gate_path = output / state["active_c0_gate_path"]
    decision_path = output / state["active_scheduling_decision_path"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert state["status"] == "C0_ENGINEERING_FAILED"
    assert state["completed_model_ids"] == []
    assert gate["status"] == "ENGINEERING_FAIL"
    assert decision["decision"] == "NO_JOBS_AFTER_C0_FAILURE"
    assert decision["queued_model_ids"] == []
    assert started == ["C-seed0"]

    resumed = campaign.run_parallel_campaign(ROOT, output, resume=True)
    assert resumed["status"] == "C0_ENGINEERING_FAILED"
    assert started == ["C-seed0"]

    retried = campaign.run_parallel_campaign(
        ROOT, output, workers=4, resume=True, retry_model_id="C-seed0"
    )
    attempts = retried["models"]["C-seed0"]["attempts"]
    assert retried["status"] == "C0_ENGINEERING_FAILED"
    assert [attempt["attempt_id"] for attempt in attempts] == ["attempt-0001", "attempt-0002"]
    assert attempts[0]["run_relative_path"] != attempts[1]["run_relative_path"]
    assert (output / attempts[0]["run_relative_path"] / "engineering-validation.json").is_file()
    assert started == ["C-seed0", "C-seed0"]
    with pytest.raises(RuntimeError, match="one-manual-retry limit"):
        campaign.run_parallel_campaign(
            ROOT, output, workers=4, resume=True, retry_model_id="C-seed0"
        )
    assert started == ["C-seed0", "C-seed0"]


def test_campaign_level_c0_pass_binds_queue_and_resume_revalidates_artifacts(tmp_path, monkeypatch):
    config, _raw, config_sha = campaign._load_campaign_config(ROOT)
    schedule = _full_test_schedule()
    base_manifest = {
        "schema_version": "f1rl_wp3_campaign_manifest_v2",
        "sha256": "base-manifest-pass-test", "source_commit": "test-source-commit",
        "source_tree_sha256": "test-source-tree-sha256",
        "campaign_config_sha256": config_sha, "training": config["training"],
    }
    parallel_config = _expected_parallel_campaign_config()
    parallel_sha = sha256((ROOT / PARALLEL_CAMPAIGN_CONFIG).read_bytes()).hexdigest()
    monkeypatch.setattr(campaign, "build_campaign_manifest",
                        lambda _root: (base_manifest, config, schedule))
    monkeypatch.setattr(campaign, "load_parallel_contract",
                        lambda _root: (parallel_config, parallel_sha))
    monkeypatch.setattr(campaign, "_parallel_preflight",
                        lambda *_args: {"logical_processors": 32, "available_memory_bytes": 40 * GIB,
                                        "host_cpu_load_percent": 2.0})
    monkeypatch.setattr(campaign, "validate_training_run", lambda _root, _run, job, *_args: {
        "schema_version": "f1rl_wp3_model_engineering_validation_v1",
        "status": "ENGINEERING_PASS", "model_id": job["model_id"],
        "learning_timesteps": 65536, "lifecycle": {
            "track_timestep_counts": {track: 1 for track in TRAIN_TRACKS},
            "automatic_resets_verified": 1,
        },
    })
    started = []

    def fake_child(_root, run_path, log_path, job, _training, **_kwargs):
        started.append(job["model_id"])
        run_path.mkdir(parents=True, exist_ok=False)
        log_path.write_text("simulated child\n", encoding="utf-8")
        result = {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.01,
                  "cleanup_seconds": 0.0, "resource_usage": {}}
        if job["model_id"] == "A-seed0":
            result["returncode"] = 7
        return result

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    import f1rl.local_demo.evaluation as evaluation
    monkeypatch.setattr(evaluation, "build_evaluation_roster", lambda *_args: {"sha256": "pass-roster"})
    output = tmp_path / "parallel-c0-pass"

    state = campaign.run_parallel_campaign(ROOT, output, workers=1)

    gate = json.loads((output / state["active_c0_gate_path"]).read_text(encoding="utf-8"))
    decision = json.loads((output / state["active_scheduling_decision_path"]).read_text(encoding="utf-8"))
    assert gate["status"] == "ENGINEERING_PASS"
    assert gate["learning_timesteps"] == 65536
    assert decision["decision"] == "OPEN_11_JOB_QUEUE"
    assert len(decision["queued_model_ids"]) == 11
    assert started == ["C-seed0", "A-seed0"]
    with pytest.raises(RuntimeError, match="C0 already passed"):
        campaign.run_parallel_campaign(ROOT, output, resume=True, retry_model_id="C-seed0")

    c0_validation_path = output / gate["run_relative_path"] / "engineering-validation.json"
    validation = json.loads(c0_validation_path.read_text(encoding="utf-8"))
    validation["learning_timesteps"] = 4096
    c0_validation_path.write_text(json.dumps(validation), encoding="utf-8")
    with pytest.raises(ValueError, match="self-checksum mismatch"):
        campaign.run_parallel_campaign(ROOT, output, resume=True)


def test_bounded_scheduler_stops_dispatch_when_global_memory_monitor_fails():
    started = []

    def child(job, *, abort_event):
        started.append(job["model_id"])
        abort_event.wait(1)
        return {"returncode": -1, "cancelled": True}

    outcome = campaign._run_bounded_stage(
        _schedule("A-seed0", "B-seed0", "C-seed0"), 2, child,
        lambda _job, _result: {"status": "ENGINEERING_PASS"},
        on_tick=lambda _active, _delta: "aggregate process-tree RSS exceeded budget",
        tick_interval_seconds=0.01,
    )

    assert outcome["status"] == "ENGINEERING_FAILED"
    assert "resource_budget" in outcome["failures"]
    assert set(started) == {"A-seed0", "B-seed0"}
    assert "C-seed0" in outcome["not_started_model_ids"]


def test_parallel_scheduler_rejects_duplicate_job_identity_before_launch():
    started = []

    with pytest.raises(ValueError, match="duplicate model identities"):
        campaign._run_bounded_stage(
            _schedule("A-seed0", "A-seed0"), 2,
            lambda job, **_kwargs: started.append(job["model_id"]),
            lambda _job, _result: {"status": "ENGINEERING_PASS"},
        )
    assert started == []


def test_parallel_source_admission_rejects_dirty_worktree(tmp_path):
    subprocess.run(["git", "init", "--quiet", "-b", "feature"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "codex@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Codex Test"], cwd=tmp_path, check=True)
    source_file = tmp_path / "source.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "test source"], cwd=tmp_path, check=True)

    commit, source_sha = _source_identity(tmp_path)
    assert len(commit) == 40
    assert len(source_sha) == 64
    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean committed source tree"):
        _source_identity(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="child process cleanup uses Windows process controls")
@pytest.mark.parametrize("failure", ["timeout", "crash", "rss"])
def test_parallel_child_timeout_crash_and_rss_failure_cleanup(tmp_path, monkeypatch, failure):
    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.returncode = 7 if failure == "crash" else None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 1

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = 1
            return self.returncode

        def kill(self):
            self.returncode = 1

    child_process = FakeProcess()

    class FakeJob:
        def __init__(self):
            self.terminated = False
            self.closed = False

        def assign(self, _pid):
            return None

        def terminate(self):
            self.terminated = True
            if child_process.returncode is None:
                child_process.returncode = 1

        def wait_empty(self, _timeout):
            return True

        def close(self):
            self.closed = True

    job = FakeJob()
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *_args, **_kwargs: child_process)
    monkeypatch.setattr(process_resources, "WindowsKillJob", lambda: job)
    if failure == "timeout":
        clock_values = iter([0.0, 122.0, 122.0, 122.0, 122.0])
        monkeypatch.setattr(campaign.time, "monotonic", lambda: next(clock_values, 122.0))
        monkeypatch.setattr(process_resources, "sample_process_tree", lambda _pid: None)
    elif failure == "crash":
        monkeypatch.setattr(process_resources, "sample_process_tree", lambda _pid: None)
    else:
        monkeypatch.setattr(process_resources, "sample_process_tree", lambda _pid: {
            "rss_bytes": 101, "process_count": 1, "cpu_seconds": 0.01,
            "processes": [{"pid": 12345, "creation_time_100ns": 1}],
        })

    log_path = tmp_path / f"{failure}.log"
    record_path = tmp_path / f"{failure}.process.json"
    model = {"model_id": "A-seed0", "candidate_id": "A", "seed_index": 0,
             "attempt_id": "attempt-0001", "source_commit": "test",
             "parallel_manifest_sha256": "test"}
    training = {"max_wall_seconds_per_model": 1}
    resources = {"memory_poll_interval_seconds": 0.01,
                 "per_worker_process_tree_rss_limit_bytes": 100,
                 "max_processes_per_worker": 1,
                 "shutdown_grace_seconds": 1, "cleanup_timeout_seconds": 1}
    if failure == "timeout":
        with pytest.raises(TimeoutError, match="wall-time guard"):
            campaign._run_parallel_child(
                ROOT, tmp_path / "run", log_path, model, training,
                abort_event=threading.Event(), resources=resources,
                process_record_path=record_path,
            )
    elif failure == "crash":
        result = campaign._run_parallel_child(
            ROOT, tmp_path / "run", log_path, model, training,
            abort_event=threading.Event(), resources=resources,
            process_record_path=record_path,
        )
        assert result["returncode"] == 7
    else:
        with pytest.raises(RuntimeError, match="RSS budget"):
            campaign._run_parallel_child(
                ROOT, tmp_path / "run", log_path, model, training,
                abort_event=threading.Event(), resources=resources,
                process_record_path=record_path,
            )
    assert job.terminated
    assert job.closed


def test_process_tree_expansion_uses_parent_identity():
    rows = [
        {"pid": 10, "parent_pid": 1}, {"pid": 11, "parent_pid": 10},
        {"pid": 12, "parent_pid": 11}, {"pid": 13, "parent_pid": 2},
    ]

    assert set(process_tree_pids(10, rows)) == {10, 11, 12}


@pytest.mark.skipif(os.name != "nt", reason="mock process benchmark uses Windows process accounting")
def test_mock_subprocess_benchmark_compares_1_2_4_without_campaign_completion(tmp_path):
    report = run_mock_subprocess_benchmark(
        tmp_path / "mock-benchmark",
        job_durations_seconds=[0.08, 0.14, 0.10, 0.18],
        memory_mib_per_job=4,
    )

    assert [row["worker_count"] for row in report["comparisons"]] == [1, 2, 4]
    assert all(row["completed_jobs"] == 4 for row in report["comparisons"])
    assert all(row["training_transitions_per_second"] is None for row in report["comparisons"])
    assert all(row["ppo_rollout_seconds"] is None and row["ppo_optimizer_seconds"] is None
               for row in report["comparisons"])
    assert all(not row["campaign_completion"] for row in report["comparisons"])
    assert all((tmp_path / "mock-benchmark" / f"workers-{worker}" /
                "raw-measurements.jsonl").is_file() for worker in (1, 2, 4))
    source_digest = sha256()
    for relative in ("f1rl/local_demo/campaign.py", "f1rl/local_demo/parallel_campaign.py",
                     "f1rl/local_demo/process_resources.py",
                     "f1rl/local_demo/parallel_benchmark.py"):
        source_digest.update(relative.encode("utf-8"))
        source_digest.update((ROOT / relative).read_bytes())
    assert report["source"]["scheduler_source_sha256"] == source_digest.hexdigest()


def _install_mock_campaign_dependencies(monkeypatch):
    config, _raw, config_sha = campaign._load_campaign_config(ROOT)
    schedule = _full_test_schedule()
    base_manifest = {
        "schema_version": "f1rl_wp3_campaign_manifest_v2",
        "sha256": "mock-base-campaign-manifest-sha256",
        "source_commit": "mock-source-commit", "source_tree_sha256": "mock-source-tree-sha256",
        "campaign_config_sha256": config_sha, "training": config["training"],
        "simulation_only_config_sha256": "mock-simulation-config-sha256",
        "registration_sha256": "mock-registration-sha256",
        "track_manifest_sha256": "mock-track-manifest-sha256",
        "execution_order": [{key: row[key] for key in ("model_id", "candidate_id", "seed_index")}
                             for row in schedule],
    }
    parallel_config = _expected_parallel_campaign_config()
    parallel_sha = sha256((ROOT / PARALLEL_CAMPAIGN_CONFIG).read_bytes()).hexdigest()
    monkeypatch.setattr(campaign, "build_campaign_manifest",
                        lambda _root: (base_manifest, config, schedule))
    monkeypatch.setattr(campaign, "load_parallel_contract",
                        lambda _root: (parallel_config, parallel_sha))
    monkeypatch.setattr(campaign, "_parallel_preflight", lambda *_args: {
        "logical_processors": 32, "available_memory_bytes": 40 * GIB,
        "host_cpu_load_percent": 2.0,
    })

    def validate_training_run(_root, run_dir, job, manifest, _config):
        telemetry_path = run_dir / "training-telemetry.jsonl"
        telemetry = telemetry_path.read_bytes()
        checkpoint_path = run_dir / "model-final.zip"
        checkpoint_sha256 = sha256(checkpoint_path.read_bytes()).hexdigest()
        return {
            "schema_version": "f1rl_wp3_model_engineering_validation_v1",
            "status": "ENGINEERING_PASS", "model_id": job["model_id"],
            "candidate_id": job["candidate_id"], "seed_index": job["seed_index"],
            "source_commit": manifest["source_commit"], "learning_timesteps": 65536,
            "optimizer_steps": 1, "initial_policy_weights_sha256": "mock-initial-weights",
            "final_policy_weights_sha256": "mock-final-weights", "weights_changed": True,
            "telemetry": {"path": telemetry_path.name, "bytes": len(telemetry),
                          "sha256": sha256(telemetry).hexdigest(), "events": 1,
                          "step_rows": 0, "update_rows": 1, "optimizer_steps": 1},
            "checkpoints": [{"path": checkpoint_path.name, "timesteps": 65536,
                              "sha256": checkpoint_sha256, "size_bytes": checkpoint_path.stat().st_size}],
            "lifecycle": {
                "episode_count": 1, "terminal_transitions": 1,
                "automatic_resets_verified": 1,
                "natural_terminations": 1, "physics_abort_terminations": 0,
                "curriculum_terminations": 0, "truncations": 0,
                "track_episode_counts": {track: 1 for track in TRAIN_TRACKS},
                "track_timestep_counts": {track: 1 for track in TRAIN_TRACKS},
            },
            "episodes": 1,
        }

    monkeypatch.setattr(campaign, "validate_training_run", validate_training_run)
    return config, schedule, base_manifest


def _write_mock_training_artifacts(run_dir: Path, model_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    telemetry = json.dumps({"event_type": "ppo_update", "metrics": {"train/loss": 2.5}}) + "\n"
    (run_dir / "training-telemetry.jsonl").write_text(telemetry, encoding="utf-8")
    checkpoint = f"mock-checkpoint:{model_id}".encode("utf-8")
    checkpoint_path = run_dir / "model-final.zip"
    checkpoint_path.write_bytes(checkpoint)
    (run_dir / "state.json").write_text(json.dumps({
        "final_model": {"path": checkpoint_path.name, "sha256": sha256(checkpoint).hexdigest()},
    }), encoding="utf-8")


def _install_mock_evaluation(monkeypatch, *, capture_checkpoints=None) -> None:
    class MockPolicy:
        num_timesteps = 65536

    monkeypatch.setattr(evaluation.PPOModel, "load", staticmethod(
        lambda path, device: (capture_checkpoints.append(Path(path)) if capture_checkpoints is not None else None)
        or MockPolicy()))
    monkeypatch.setattr(evaluation, "replace", lambda value, **changes: SimpleNamespace(
        **{**vars(value), **changes}))
    monkeypatch.setattr(evaluation, "_evaluation_factories", lambda _root, _protocol, _seed, tracks, strata: {
        (track, stratum): SimpleNamespace(
            race_config=SimpleNamespace(hard_corridor_m=4.0),
            reset_config=SimpleNamespace(seed=123),
            simulation_config=SimpleNamespace(fixed_dt=0.02),
        ) for track in tracks for stratum in strata
    })
    def evaluate_mock(_policy, _factory, _track_id, environment_seed, policy_seed, _fixed_dt):
        return {
            "success": True, "final_normalized_route_progress": 1.0,
            "completed_checkpoints": 4, "collision_count": 0,
            "hard_route_violation": False, "timeout": False,
            "steering_rate_rms_rad_per_s": 0.0, "episode_time": 1.0,
            "failure_reason": None, "environment_seed": environment_seed,
            "policy_seed": policy_seed,
        }

    monkeypatch.setattr(evaluation, "_evaluate_episode", evaluate_mock)


def test_full_mock_parallel_campaign_evaluates_attempts_writes_reports_and_resumes(tmp_path, monkeypatch):
    _install_mock_campaign_dependencies(monkeypatch)
    loaded_checkpoints = []
    _install_mock_evaluation(monkeypatch, capture_checkpoints=loaded_checkpoints)
    started = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_child(_root, run_dir, log_path, job, _training, *, abort_event, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            started.append(job["model_id"])
        try:
            _write_mock_training_artifacts(run_dir, job["model_id"])
            log_path.write_text("mock child passed\n", encoding="utf-8")
            abort_event.wait(0.002)
            return {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.002,
                    "cleanup_seconds": 0.0, "resource_usage": {}}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    output = tmp_path / "full-mock-campaign"

    state = campaign.run_parallel_campaign(ROOT, output, workers=2)

    assert state["status"] == "COMPLETE"
    assert len(started) == 12 and len(set(started)) == 12
    assert started[0] == "C-seed0" and max_active <= 2
    assert all(entry["attempts"][0]["run_relative_path"] ==
               f"runs/{model_id}/attempt-0001" for model_id, entry in state["models"].items())
    assert len(loaded_checkpoints) == 12
    assert all(path.parent.as_posix().endswith(f"{path.parent.parent.name}/attempt-0001")
               for path in loaded_checkpoints)
    assert state["evaluation"]["status"] == "PASS"
    assert (output / "WP3-LEARNING-PROGRESS.svg").is_file()
    status_doc = json.loads((output / "WP3-PARALLEL-REVIEW-STATUS.json").read_text(encoding="utf-8"))
    assert status_doc["engineering_status"] == "ENGINEERING_VALIDATION_COMPLETE"
    assert status_doc["independent_review_status"] == "PENDING_INDEPENDENT_REVIEW"
    assert not (output / "WP3-INDEPENDENT-REVIEW.md").exists()
    campaign._verify_evidence_manifest(output)
    manifest = json.loads((output / "WP3-EVIDENCE-MANIFEST.json").read_text(encoding="utf-8"))
    assert "WP3-LEARNING-PROGRESS.svg" in {row["path"] for row in manifest["files"]}
    assert "runs/A-seed0/attempt-0001/training-telemetry.jsonl" in {
        row["path"] for row in manifest["files"]
    }

    resumed = campaign.run_parallel_campaign(ROOT, output, resume=True)

    assert resumed["status"] == "COMPLETE"
    assert len(started) == 12
    with pytest.raises(RuntimeError, match="fully validated parallel job"):
        campaign.run_parallel_campaign(ROOT, output, resume=True, retry_model_id="A-seed0")
    assert len(started) == 12
    with pytest.raises(RuntimeError, match="fully validated parallel job"):
        campaign.run_parallel_campaign(ROOT, output, resume=True, retry_model_id="A-seed0")
    assert len(started) == 12


def test_multiple_failed_jobs_can_be_retried_one_at_a_time_before_queue_resumes(tmp_path, monkeypatch):
    _install_mock_campaign_dependencies(monkeypatch)
    monkeypatch.setattr(evaluation, "run_evaluation", lambda *_args: {
        "status": "PASS", "scientific_evidence": False, "heldout_access": False,
    })
    started = []

    def fake_child(_root, run_dir, log_path, job, _training, *, abort_event, **_kwargs):
        model_id, attempt_id = job["model_id"], job["attempt_id"]
        started.append((model_id, attempt_id))
        if attempt_id == "attempt-0001" and model_id in {"A-seed0", "B-seed0"}:
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path.write_text("preserved failed attempt\n", encoding="utf-8")
            if model_id == "B-seed0":
                abort_event.wait(1)
                return {"returncode": -1, "cancelled": True, "elapsed_seconds": 0.01}
            return {"returncode": 9, "cancelled": False, "elapsed_seconds": 0.01}
        _write_mock_training_artifacts(run_dir, model_id)
        log_path.write_text("mock child passed\n", encoding="utf-8")
        return {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.01,
                "cleanup_seconds": 0.0, "resource_usage": {}}

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    output = tmp_path / "retry-mock-campaign"
    failed = campaign.run_parallel_campaign(ROOT, output, workers=2)
    assert failed["models"]["A-seed0"]["status"] == "FAILED"
    assert failed["models"]["B-seed0"]["status"] == "CANCELLED"
    assert len(started) == 3 and "D-seed0" not in {model for model, _ in started}
    with pytest.raises(RuntimeError, match="explicit retries"):
        campaign.run_parallel_campaign(ROOT, output, workers=2, resume=True)
    assert len(started) == 3

    retried_a = campaign.run_parallel_campaign(
        ROOT, output, workers=2, resume=True, retry_model_id="A-seed0")
    assert retried_a["status"] == "WAITING_FOR_EXPLICIT_RETRY"
    assert retried_a["recovery"]["status"] == "UNRESOLVED_FAILURES_REMAIN"
    assert len(started) == 4 and started[-1] == ("A-seed0", "attempt-0002")
    assert retried_a["models"]["A-seed0"]["attempts"][0]["status"] == "FAILED"
    assert retried_a["models"]["A-seed0"]["attempts"][1]["status"] == "ENGINEERING_PASS"
    assert not any(model not in {"C-seed0", "A-seed0", "B-seed0"} for model, _ in started)

    with pytest.raises(RuntimeError, match="fully validated parallel job"):
        campaign.run_parallel_campaign(
            ROOT, output, workers=2, resume=True, retry_model_id="A-seed0")
    resumed = campaign.run_parallel_campaign(
        ROOT, output, workers=2, resume=True, retry_model_id="B-seed0")
    assert resumed["status"] == "COMPLETE"
    assert resumed["recovery"]["status"] == "SAFE_TO_RESUME_PENDING_QUEUE"
    assert started.count(("A-seed0", "attempt-0001")) == 1
    assert started.count(("A-seed0", "attempt-0002")) == 1
    assert started.count(("B-seed0", "attempt-0001")) == 1
    assert started.count(("B-seed0", "attempt-0002")) == 1
    assert len(started) == 14
    assert (output / "runs/A-seed0/attempt-0001").is_dir()
    assert (output / "runs/A-seed0/attempt-0002").is_dir()
    assert (output / "runs/B-seed0/attempt-0001").is_dir()
    assert (output / "runs/B-seed0/attempt-0002").is_dir()


def test_c0_retry_must_pass_full_gate_before_opening_queue(tmp_path, monkeypatch):
    _install_mock_campaign_dependencies(monkeypatch)
    validator = campaign.validate_training_run

    def validate_with_c0_attempt_lifecycle(root, run_dir, job, manifest, config):
        result = validator(root, run_dir, job, manifest, config)
        if job["model_id"] == "C-seed0" and run_dir.name == "attempt-0001":
            result["lifecycle"]["automatic_resets_verified"] = 0
        return result

    monkeypatch.setattr(campaign, "validate_training_run", validate_with_c0_attempt_lifecycle)
    monkeypatch.setattr(evaluation, "run_evaluation", lambda *_args: {
        "status": "PASS", "scientific_evidence": False, "heldout_access": False,
    })
    started = []

    def fake_child(_root, run_dir, log_path, job, _training, **_kwargs):
        if job["model_id"] != "C-seed0":
            gate_path = sorted((output / "gates").glob("C0-ENGINEERING-GATE-*.json"))[-1]
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            assert gate["status"] == "ENGINEERING_PASS"
            assert gate["learning_timesteps"] == 65536
        started.append((job["model_id"], job.get("attempt_id")))
        _write_mock_training_artifacts(run_dir, job["model_id"])
        log_path.write_text("mock child passed\n", encoding="utf-8")
        return {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.01,
                "cleanup_seconds": 0.0, "resource_usage": {}}

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    output = tmp_path / "c0-retry-campaign"

    failed = campaign.run_parallel_campaign(ROOT, output, workers=2)
    assert failed["status"] == "C0_ENGINEERING_FAILED"
    assert started == [("C-seed0", "attempt-0001")]

    passed = campaign.run_parallel_campaign(
        ROOT, output, workers=2, resume=True, retry_model_id="C-seed0")

    assert passed["status"] == "COMPLETE"
    assert passed["recovery"]["status"] == "SAFE_TO_RESUME_PENDING_QUEUE"
    assert len(started) == 13 and started[0] == ("C-seed0", "attempt-0001")
    assert started[1] == ("C-seed0", "attempt-0002")
    assert len({model_id for model_id, _attempt in started[1:]}) == 12
    gate = json.loads((output / passed["active_c0_gate_path"]).read_text(encoding="utf-8"))
    assert gate["status"] == "ENGINEERING_PASS"
    assert gate["run_relative_path"] == "runs/C-seed0/attempt-0002"


def test_failed_retry_keeps_pending_queue_closed_and_retry_budget_bounded(tmp_path, monkeypatch):
    _install_mock_campaign_dependencies(monkeypatch)
    started = []

    def fake_child(_root, run_dir, log_path, job, _training, *, abort_event, **_kwargs):
        model_id, attempt_id = job["model_id"], job["attempt_id"]
        started.append((model_id, attempt_id))
        if attempt_id == "attempt-0001" and model_id == "A-seed0":
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path.write_text("initial failure\n", encoding="utf-8")
            return {"returncode": 9, "cancelled": False, "elapsed_seconds": 0.01}
        if attempt_id == "attempt-0001" and model_id == "B-seed0":
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path.write_text("cancelled sibling\n", encoding="utf-8")
            abort_event.wait(1)
            return {"returncode": -1, "cancelled": True, "elapsed_seconds": 0.01}
        if model_id == "A-seed0" and attempt_id == "attempt-0002":
            run_dir.mkdir(parents=True, exist_ok=False)
            log_path.write_text("retry failure\n", encoding="utf-8")
            return {"returncode": 8, "cancelled": False, "elapsed_seconds": 0.01}
        _write_mock_training_artifacts(run_dir, model_id)
        log_path.write_text("mock child passed\n", encoding="utf-8")
        return {"returncode": 0, "cancelled": False, "elapsed_seconds": 0.01,
                "cleanup_seconds": 0.0, "resource_usage": {}}

    monkeypatch.setattr(campaign, "_run_child", fake_child)
    output = tmp_path / "failed-retry-campaign"
    campaign.run_parallel_campaign(ROOT, output, workers=2)
    retried = campaign.run_parallel_campaign(
        ROOT, output, workers=2, resume=True, retry_model_id="A-seed0")

    assert retried["status"] == "ENGINEERING_FAILED"
    assert retried["recovery"]["status"] == "RETRY_FAILED"
    assert started == [("C-seed0", "attempt-0001"),
                       ("A-seed0", "attempt-0001"), ("B-seed0", "attempt-0001"),
                       ("A-seed0", "attempt-0002")]
    assert retried["models"]["A-seed0"]["attempts"][0]["status"] == "FAILED"
    assert retried["models"]["A-seed0"]["attempts"][1]["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="one-manual-retry limit"):
        campaign.run_parallel_campaign(
            ROOT, output, workers=2, resume=True, retry_model_id="A-seed0")
    assert len(started) == 4


def test_parallel_retry_rejects_a_still_live_child(tmp_path, monkeypatch):
    identity_path = tmp_path / "attempt-meta" / "A-seed0" / "attempt-0001" / "process-identity.json"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(json.dumps({
        "model_id": "A-seed0", "attempt_id": "attempt-0001",
        "pid": 123, "creation_time_100ns": 456,
    }), encoding="utf-8")
    monkeypatch.setattr(process_resources, "sample_process_tree", lambda _pid: {
        "processes": [{"pid": 123, "creation_time_100ns": 456}],
    })
    state = {"models": {"A-seed0": {"attempts": [{"attempt_id": "attempt-0001"}]}}}

    with pytest.raises(RuntimeError, match="still alive"):
        campaign._parallel_assert_no_live_children(tmp_path, state)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior is platform-specific")
def test_windows_job_cleanup_terminates_fake_grandchild(tmp_path):
    start_path = tmp_path / "start-child"
    child_pid_path = tmp_path / "grandchild.pid"
    parent_code = (
        "from pathlib import Path; import subprocess,sys,time; "
        "start=Path(sys.argv[1]); target=Path(sys.argv[2]); "
        "while_not_started=True\n"
        "while not start.exists(): time.sleep(0.01)\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "target.write_text(str(child.pid),encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(start_path), str(child_pid_path)],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    job = WindowsKillJob()
    try:
        job.assign(parent.pid)
        start_path.write_text("go", encoding="utf-8")
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists()
        grandchild_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert grandchild_pid in process_tree_pids(parent.pid)
        job.terminate()
        parent.wait(timeout=5)
        assert job.wait_empty(5)
        deadline = time.monotonic() + 5
        while grandchild_pid in {row["pid"] for row in _process_rows()} and time.monotonic() < deadline:
            time.sleep(0.05)
        assert grandchild_pid not in {row["pid"] for row in _process_rows()}
    finally:
        if parent.poll() is None:
            job.terminate()
            parent.wait(timeout=5)
        job.close()
