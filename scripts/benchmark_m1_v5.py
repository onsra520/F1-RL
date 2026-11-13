"""Measure one development-only custom Auto Drive route under Layered Physics V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from f1rl.local_demo.runtime.catalog import RuntimeCatalog
from f1rl.local_demo.runtime.custom_tracks import convert_custom_track
from f1rl.local_demo.runtime.session import RuntimeSessionManager
from f1rl.runtime.simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/runtime/fixtures/m1-development-geometry-v1.json"
PHYSICS_PROFILE = "experimental_v5_layered"
SEED = 412


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256() -> str:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, capture_output=True,
    ).stdout.split(b"\0")
    digest = hashlib.sha256()
    for raw_path in sorted(path for path in listed if path):
        relative = Path(os.fsdecode(raw_path))
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(hashlib.sha256((ROOT / relative).read_bytes()).digest())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture = next(row for row in fixture_data["fixtures"]
                   if row["id"] == "m1-intersection-overpass-v1")
    catalog = RuntimeCatalog(ROOT)
    converted = convert_custom_track(fixture["track"], catalog)
    physics_steps: list[tuple[float, float]] = []
    original_step = Simulation.step

    def measured_step(simulation, control):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            return original_step(simulation, control)
        finally:
            physics_steps.append((time.perf_counter() - wall_start,
                                 time.process_time() - cpu_start))

    manager = RuntimeSessionManager(catalog)
    Simulation.step = measured_step
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        session = manager.create(
            model_id=None,
            track_id=converted.track_id,
            seed=SEED,
            mode="auto",
            custom_factory=converted.factory,
            custom_track_sha256=converted.track_sha256,
            physics_profile=PHYSICS_PROFILE,
        )
        session.start()
        completed = session.wait(600.0)
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        status = session.status_document()
        result = status.get("episode_result") or {}
        if not completed or status["status"] != "finished" or not result.get("successful_finish"):
            raise RuntimeError(f"V5 benchmark route did not finish: {status}")
        wall_steps = [row[0] for row in physics_steps]
        cpu_steps = [row[1] for row in physics_steps]
        sim_seconds = float(result["duration_s"])
        output = {
            "schema": "f1rl_m1_v5_benchmark_v1",
            "measured_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
            "source_tree_sha256": source_tree_sha256(),
            "fixture_id": fixture["id"],
            "fixture_sha256": sha256(FIXTURE),
            "physics_profile": PHYSICS_PROFILE,
            "physics_config_sha256": status["physics"]["sha256"],
            "seed": SEED,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "ticks": len(physics_steps),
            "simulation_seconds": sim_seconds,
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "realtime_factor": sim_seconds / wall_seconds if wall_seconds else None,
            "physics_step_wall_mean_ms": statistics.mean(wall_steps) * 1000,
            "physics_step_wall_p95_ms": float(np.percentile(wall_steps, 95)) * 1000,
            "physics_step_cpu_mean_ms": statistics.mean(cpu_steps) * 1000,
            "physics_step_cpu_p95_ms": float(np.percentile(cpu_steps, 95)) * 1000,
            "successful_finish": result["successful_finish"],
            "collision_count": sum(frame["collision_count"]
                                    for frame in session.replay()["frames"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
        return 0
    finally:
        Simulation.step = original_step
        manager.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
