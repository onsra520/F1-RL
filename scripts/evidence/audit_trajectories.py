"""Audit stored WP7 real PPO runs, without changing original model evidence.

Use --run for calibration and confirmation. Writes a NEW audit file outside
source worktree and does not access HELD-OUT tracks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.validation.trajectory_audit_cli import audit_trajectory_run, main as _main

def main()->int:
    return _main(ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
