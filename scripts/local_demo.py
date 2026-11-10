from __future__ import annotations

import argparse
from pathlib import Path
import sys

from f1rl.local_demo.runner import run_local_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded LOCAL_DEMO training job")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--candidate", choices=("A", "B", "C", "D"), required=True)
    train.add_argument("--tuning-seed-index", type=int, choices=(0, 1, 2), required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--profile", choices=("tuning", "smoke", "benchmark"), default="tuning")
    args = parser.parse_args(argv)
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path")

    repository_root = Path(__file__).resolve().parents[1]
    try:
        result = run_local_demo(
            repository_root,
            candidate_id=args.candidate,
            tuning_seed_index=args.tuning_seed_index,
            output_directory=args.output,
            dry_run=False,
            profile=args.profile,
        )
    except Exception as error:
        print(f"[LOCAL_DEMO_CHILD] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1

    state = result.get("state") if isinstance(result, dict) else None
    expected_status = "SMOKE_INTEGRATION_PASS" if args.profile == "smoke" else "learning_completed"
    if not isinstance(state, dict) or state.get("status") != expected_status:
        print("[LOCAL_DEMO_CHILD] runner did not report learning_completed", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
