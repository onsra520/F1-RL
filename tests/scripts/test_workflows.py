"""Workflow routing is complete and cannot weaken historical entrypoints."""
import json
from pathlib import Path
import subprocess
import sys

from scripts.workflows.catalog import WORKFLOWS
from scripts.workflows.__main__ import BY_COMMAND, ROOT, main, manifest


def test_all_original_script_entrypoints_are_catalogued():
    original = {
        path.name for path in (ROOT / "scripts").glob("*.py")
        if path.name != "__init__.py"
    }
    assert len(original) == 48
    assert {row.script for row in WORKFLOWS} == original
    assert len(BY_COMMAND) == len(WORKFLOWS)


def test_workflow_names_are_versionless_and_unambiguous():
    import re
    for row in WORKFLOWS:
        assert re.fullmatch(r"[a-z][a-z0-9-]*", row.group)
        assert re.fullmatch(r"[a-z][a-z0-9-]*", row.name)
        assert not re.search(r"(?:^|-)wp\d+(?:-|$)|(?:^|-)v\d+(?:-|$)", row.name)
    assert len({(row.group, row.name) for row in WORKFLOWS}) == len(WORKFLOWS)


def test_catalogue_is_non_executing_and_machine_readable(capsys):
    assert main(["list", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result) == 48
    assert result == manifest()


def test_invalid_command_fails_without_execution(capsys):
    assert main(["does-not-exist", "train"]) == 2
    assert main(["internal", "ppo-fixture-worker"]) == 2
    assert main(["evaluation"]) == 2


def test_module_entrypoint_reports_help_without_running_workflows():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.workflows", "list", "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=20, check=True,
    )
    assert len(json.loads(result.stdout)) == 48
