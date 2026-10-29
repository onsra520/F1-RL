"""Functional script package invariants and stable legacy compatibility."""
from pathlib import Path
import json
import re
from scripts.workflows.catalog import WORKFLOWS, PRIMARY_PATHS

ROOT = Path(__file__).resolve().parents[2]


def test_grouped_implementation_paths_are_complete():
    registry = json.loads((ROOT / "docs/development/SCRIPT-STATUS.json").read_text("utf-8"))
    rows = {(item.group, item.name): item for item in WORKFLOWS}
    assert len(rows) == 48
    grouped = registry["workflows"]
    extracted = [item for item in grouped if item["refactor_state"] == "implementation-extracted"]
    assert len(extracted) == len(PRIMARY_PATHS) == 14
    for row in grouped:
        key = (row["group"], row["alias"])
        assert key in rows
        assert (ROOT / row["legacy_path"]).exists()
        primary = ROOT / row["primary_entrypoint"]
        assert primary.is_file()
        if row in extracted:
            assert row["primary_entrypoint"] == PRIMARY_PATHS[key]
            assert primary.parent.name == row["group"]
            assert re.fullmatch(r"[a-z][a-z0-9_]*", primary.stem)
            assert (primary.parent / "__init__.py").is_file()
            assert "from src." in primary.read_text("utf-8")
            legacy = (ROOT / row["legacy_path"]).read_text("utf-8")
            assert f"from scripts.{row['group']}" in legacy
            assert "def __getattr__" in legacy
        else:
            assert key not in PRIMARY_PATHS
            assert row["primary_entrypoint"] == row["legacy_path"]


def test_primary_groups_are_import_safe():
    import importlib
    for group in ("diagnostics", "evaluation", "evidence", "inference", "training"):
        module = importlib.import_module("scripts." + group)
        assert not hasattr(module, "main")
