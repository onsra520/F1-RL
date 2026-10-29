"""Explicit Windows-only simulation research demo bootstrap; no scientific permit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.interfaces.local_demo_cli import main as _main

def main():
    return _main(ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
