"""Replay only the archived Right Turn TRAIN final PPO models for road-union audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.validation.road_corridor_audit_cli import audit_right_turn_corridor, main as _main

def main()->int:
    return _main(ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
