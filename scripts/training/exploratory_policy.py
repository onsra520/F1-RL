"""CPU-only executable bootstrap; CLI preflight never starts a pilot."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from src.exploratory.cli import main
    raise SystemExit(main())
