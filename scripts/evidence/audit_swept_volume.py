"""Read-only WP7 3D swept body-proxy sensitivity audit for stored TRAIN PPO runs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.validation.swept_volume_audit_cli import audit_3d_swept_run, main as _main

def main()->int:
    return _main(ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
