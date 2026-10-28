"""Explicit Windows-only simulation research demo bootstrap; no scientific permit."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    if os.name != "nt":
        raise SystemExit("LOCAL_DEMO supports local Windows only")
    if sys.version_info < (3, 11):
        raise SystemExit("F1-RL requires Python 3.11+")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    parser = argparse.ArgumentParser(description="Simulation-only F1-RL local PPO demo")
    parser.add_argument("command", choices=("dry-run", "train", "smoke"))
    parser.add_argument("--candidate", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--tuning-seed-index", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "smoke" and args.output is None:
        parser.error("smoke requires a fresh absolute external --output directory")
    if args.output is None:
        project_root = ROOT.parents[2]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = project_root / ".evidence" / "F1-RL" / "local-demo" / f"{run_id}-{args.candidate}-tune{args.tuning_seed_index}"
    else:
        output = args.output
    from src.interfaces.runner import run_local_demo

    def progress(state):
        print(f"[LOCAL_DEMO] {state['phase']} timesteps={state['learning_timesteps']} "
              f"optimizer_steps={state['optimizer_steps']}", flush=True)

    result = run_local_demo(ROOT, candidate_id=args.candidate,
                            tuning_seed_index=args.tuning_seed_index,
                            output_directory=output, dry_run=args.command == "dry-run",
                            progress_observer=progress if args.command in {"train", "smoke"} else None,
                            profile="smoke" if args.command == "smoke" else "tuning")
    print(f"[LOCAL_DEMO] {result['state']['status']} output={output}", flush=True)
    if args.command == "smoke":
        print("[LOCAL_DEMO] classification=LOCAL_DEMO_SMOKE_NOT_SCIENTIFIC", flush=True)
    print("[LOCAL_DEMO] Results are simulation-only and do not satisfy WP7 scientific acceptance.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
