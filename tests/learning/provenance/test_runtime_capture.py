from importlib import metadata
from pathlib import Path

import pytest

from src.rl.provenance import _capture_runtime_provenance


EXPECTED = (
    "numpy",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "PyYAML",
)


def test_runtime_capture_uses_fixed_package_lookup_order() -> None:
    calls: list[str] = []

    def lookup(name: str) -> str:
        calls.append(name)
        return f"{name}-version"

    runtime = _capture_runtime_provenance(version_lookup=lookup)

    assert tuple(item.name for item in runtime.packages) == EXPECTED
    assert tuple(item.version for item in runtime.packages) == tuple(
        f"{name}-version" for name in EXPECTED
    )
    assert tuple(calls) == EXPECTED
    assert runtime.python_version
    assert runtime.python_implementation
    assert runtime.system
    assert runtime.release
    assert runtime.machine


def test_runtime_capture_rejects_missing_required_distribution() -> None:
    def lookup(name: str) -> str:
        if name == "torch":
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    with pytest.raises(RuntimeError, match="torch"):
        _capture_runtime_provenance(version_lookup=lookup)


@pytest.mark.parametrize("bad_version", ["", "   ", None, 123])
def test_runtime_capture_rejects_invalid_version_text(bad_version: object) -> None:
    def lookup(name: str) -> str:
        return bad_version if name == "numpy" else "1.0"  # type: ignore[return-value]

    with pytest.raises(RuntimeError, match="numpy"):
        _capture_runtime_provenance(version_lookup=lookup)


def test_real_runtime_capture_matches_installed_distribution_versions() -> None:
    runtime = _capture_runtime_provenance()
    assert tuple(item.name for item in runtime.packages) == EXPECTED
    assert tuple(item.version for item in runtime.packages) == tuple(
        metadata.version(name) for name in EXPECTED
    )


def test_provenance_module_does_not_import_frameworks_for_versions() -> None:
    root = Path(__file__).parents[3]
    source = (root / "src" / "rl" / "provenance.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "from stable_baselines3",
        "import stable_baselines3",
        "from torch",
        "import torch",
        "from gymnasium",
        "import gymnasium",
        "from src.rl.algorithms",
        "from src.training.vector_env",
        "from .algorithms",
        "from ..algorithms",
        "from .training.vector_env",
        "from ..training.vector_env",
    )
    for text in forbidden:
        assert text not in source
