"""Compatibility of extracted demo/packaging CLIs without starting training."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("script,group,action,expected",[
    ("build_demo_handoff.py","inference","package-handoff",("--run","--destination")),
    ("run_inference.py","inference","run-policy",("--episodes","--model")),
    ("local_demo.py","training","local-simulation",("--candidate","--tuning-seed-index")),
])
def test_cli_help_parity(script,group,action,expected):
    old=subprocess.run([sys.executable,str(ROOT/"scripts"/script),"--help"],
                       cwd=ROOT,capture_output=True,text=True,timeout=30)
    modern=subprocess.run([sys.executable,"-m","scripts.workflows",group,action,"--help"],
                          cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert old.returncode==modern.returncode==0,(old.stderr,modern.stderr)
    assert old.stdout==modern.stdout
    assert all(option in old.stdout for option in expected)


def test_inference_rejects_invalid_episode_count():
    result=subprocess.run([sys.executable,str(ROOT/"scripts/run_inference.py"),
                           "--episodes","0"],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert "episodes must be between 1 and 32" in result.stderr


def test_demo_compatibility_imports_and_root():
    from scripts import build_demo_handoff, local_demo, run_inference
    assert build_demo_handoff.ROOT==local_demo.ROOT==run_inference.ROOT==ROOT
    assert callable(build_demo_handoff.save)
    assert callable(build_demo_handoff.main)
    assert callable(local_demo.main)
    assert callable(run_inference.main)


def test_handoff_rejects_missing_inputs_without_outputs(tmp_path):
    from src.interfaces.demo_handoff import main
    dest=tmp_path/"package"
    with pytest.raises(FileNotFoundError):
        main(ROOT,["--run",str(tmp_path/"missing"),"--destination",str(dest)],
             torch_factory=lambda: pytest.fail("invalid input must not allocate Torch"))
    assert not dest.exists()


def test_demo_handoff_copies_grouped_inference_runtime(tmp_path):
    import shutil
    from src.interfaces.demo_handoff import demo_runtime_scripts

    portable = tmp_path / "demo"
    portable.mkdir()
    # A packaged handoff includes Python sources and their package init files.
    shutil.copytree(ROOT / "src", portable / "src", ignore=shutil.ignore_patterns("__pycache__"))
    members = demo_runtime_scripts(ROOT)
    assert len(members) == 5 and all(path.is_file() for path in members)
    for path in members:
        dest = portable / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    command = [sys.executable, str(portable / "scripts/run_inference.py"), "--help"]
    check = subprocess.run(command, cwd=portable, capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stderr
    assert "--model" in check.stdout
