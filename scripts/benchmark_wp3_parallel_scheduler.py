from __future__ import annotations

import argparse
from pathlib import Path

from f1rl.local_demo.parallel_benchmark import run_mock_subprocess_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the WP3 scheduler with mock subprocesses only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durations", type=float, nargs="+")
    parser.add_argument("--memory-mib-per-job", type=int, default=8)
    args = parser.parse_args(argv)
    report = run_mock_subprocess_benchmark(
        args.output, job_durations_seconds=args.durations,
        memory_mib_per_job=args.memory_mib_per_job,
    )
    print(f"[WP3-MOCK-BENCHMARK] runs={len(report['comparisons'])} "
          f"campaign_completion={report['campaign_completion']} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
