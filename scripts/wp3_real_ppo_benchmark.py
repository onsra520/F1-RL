from __future__ import annotations

import argparse
from pathlib import Path
import sys

from f1rl.local_demo.real_ppo_benchmark import run_real_ppo_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the gated real PPO smoke and A/B benchmark")
    parser.add_argument("mode", choices=("smoke", "sequential", "parallel"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output_root.is_absolute():
        parser.error("--output-root must be an absolute path")
    root = Path(__file__).resolve().parents[1]
    try:
        result = run_real_ppo_benchmark(root, args.output_root, args.mode)
    except Exception as error:
        print(f"[WP3_REAL_PPO_BENCHMARK] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(result, flush=True)
    return 0 if result.get("status") == "ENGINEERING_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
