"""Regression of extracted CLI implementations and the original script paths."""
from pathlib import Path
import subprocess
import sys

import pytest

from src.validation.frozen_asset_integrity import verify_frozen_assets

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_assets_are_unchanged_and_module_returns_counts():
    assert verify_frozen_assets(ROOT) == {"models": 12, "reports": 7, "cell_logs": 96}


def test_frozen_verifier_refuses_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_frozen_assets(tmp_path)


def test_legacy_frozen_verifier_matches_public_workflow():
    old = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_consolidated_wp3_assets.py")],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=120,
    )
    new = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", "evidence", "verify-frozen-assets"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=120,
    )
    assert old.stdout == new.stdout
    assert old.stdout.strip() == "PASS: 12 models, 7 reports and 96 cell logs"


def test_vector_benchmark_cli_help_is_nonexecuting():
    entrypoints = [
        [sys.executable, "scripts/benchmark_vector_envs.py", "--help"],
        [sys.executable, "-m", "scripts.workflows", "diagnostics", "benchmark-environments", "--help"],
    ]
    for command in entrypoints:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "--n-envs" in result.stdout
        assert "--transitions" in result.stdout


def test_vector_benchmark_parser_rejects_nonpositive_counts():
    from src.training.vector_benchmark import _parser
    parser = _parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--backend", "dummy", "--n-envs", "0"])
    assert error.value.code == 2
