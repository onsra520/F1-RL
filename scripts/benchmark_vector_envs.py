"""Compatibility entrypoint for the versionless vector benchmark implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.vector_benchmark import (
    _factory, _measure, _parser, _positive_int, main,
)


if __name__ == "__main__":
    main()
