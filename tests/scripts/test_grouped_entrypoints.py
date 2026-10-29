"""Functional script package invariants and stable workflow routing."""
from pathlib import Path
import json
import re

from scripts.workflows.catalog import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]


def test_all_workflows_use_grouped_modules_without_root_compatibility_files():
    status = json.loads((ROOT / "docs/development/SCRIPT-STATUS.json").read_text("utf-8"))
    rows = {(item.group, item.name): item for item in WORKFLOWS}
    assert len(rows) == 46
    assert status["counts"]["grouped_entrypoints"] == 46
    assert status["counts"]["root_legacy_entrypoints"] == 0
    assert not [path for path in (ROOT / "scripts").glob("*.py") if path.name != "__init__.py"]
    for row in status["workflows"]:
        key = (row["group"], row["alias"])
        assert key in rows
        entrypoint = ROOT / row["entrypoint"]
        assert entrypoint.is_file()
        assert entrypoint.parent.name == row["group"]
        assert re.fullmatch(r"[a-z][a-z0-9_]*", entrypoint.stem)
        assert (entrypoint.parent / "__init__.py").is_file()
        source = entrypoint.read_text("utf-8")
        assert "sys.path.insert" not in source
        assert "sys.path.append" not in source
        assert "from scripts." not in source or row["group"] == "workflows"
        assert not (ROOT / row["former_root_path"]).exists()


def test_primary_groups_are_import_safe():
    import importlib
    for group in ("diagnostics", "evaluation", "evidence", "inference", "training", "internal"):
        module = importlib.import_module("scripts." + group)
        assert not hasattr(module, "main")
