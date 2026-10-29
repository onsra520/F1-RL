"""Compatibility entrypoint for the versionless vector benchmark implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from src.training.vector_benchmark import (
    _factory, _measure, _parser, _positive_int, main,
)


if __name__ == "__main__":
    main()
