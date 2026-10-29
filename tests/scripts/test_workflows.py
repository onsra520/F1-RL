"""Workflow routing is complete and cannot weaken historical entrypoints."""
import json
import subprocess
import sys

from scripts.workflows.catalog import WORKFLOWS
from scripts.workflows.__main__ import BY_COMMAND, ROOT, main, manifest


def test_all_grouped_entrypoints_are_catalogued():
    root_entries = [path for path in (ROOT / "scripts").glob("*.py") if path.name != "__init__.py"]
    assert root_entries == []
    assert len(WORKFLOWS) == len(BY_COMMAND) == 46
    assert {row.script for row in WORKFLOWS} == {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "scripts").rglob("*.py")
        if path.name != "__init__.py" and "workflows" not in path.parts
    }


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
    assert len(result) == 46
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
    assert len(json.loads(result.stdout)) == 46


def test_dispatch_preserves_interpreter_arguments_and_exit_status(monkeypatch):
    import importlib
    cli = importlib.import_module("scripts.workflows.__main__")
    calls = []

    def fake_call(command, **kwargs):
        calls.append((command, kwargs))
        return 17

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["evidence", "verify-frozen-assets", "--some-arg", "value"]) == 17
    assert calls == [([
        sys.executable, "-m", "scripts.evidence.verify_frozen_assets",
        "--some-arg", "value",
    ], {"cwd": ROOT})]


def test_document_navigation_links_and_baseline_inventory():
    import re
    docs = (
        "README.md",
        "scripts/README.md",
        "docs/README.md",
        "docs/development/README.md",
        "docs/development/WORKFLOWS.md",
        "docs/development/SCRIPT-MIGRATION-MAP.md",
        "docs/operations/README.md",
        "docs/operations/HISTORICAL-RECOVERY.md",
        "docs/research/README.md",
        "docs/superpowers/README.md",
        "docs/architecture/README.md",
    )
    for relative in docs:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^]]+\]\(([^)]+)\)", text):
            link = match.group(1).split("#")[0]
            if not link or "://" in link or link.startswith("mailto:"):
                continue
            assert (path.parent / link).exists(), (relative, link)
    inventory = json.loads((ROOT / "docs/BASELINE-DOCUMENT-INVENTORY.json").read_text("utf-8"))
    assert inventory["total_docs"] == 324
    assert len(inventory["files"]) == inventory["total_docs"]
    assert len({entry["path"] for entry in inventory["files"]}) == 324
    assert all((ROOT / entry["path"]).is_file() for entry in inventory["files"])
