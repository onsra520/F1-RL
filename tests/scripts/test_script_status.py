"""Keep migration status aligned with actual workflow catalogue and module ownership."""
import json
from pathlib import Path

from scripts.workflows.catalog import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]


def test_migration_status_inventory_covers_every_original_workflow():
    status = json.loads((ROOT / "docs/development/SCRIPT-STATUS.json").read_text(encoding="utf-8"))
    mapping = {entry["legacy_path"]: entry for entry in status["workflows"]}
    assert len(mapping) == len(status["workflows"]) == len(WORKFLOWS) == 48
    for entry in WORKFLOWS:
        old = "scripts/" + entry.script
        record = mapping[old]
        assert (record["group"], record["alias"], record["scope"]) == (
            entry.group, entry.name, entry.scope
        )
        assert (ROOT / old).is_file()
        if record["module_owner"]:
            assert (ROOT / record["module_owner"]).is_file()
    assert status["counts"] == {
        "implementation-extracted": 8,
        "shared-dependency-migrated": 3,
        "compatibility-unmodified": 37,
    }
    assert not any(entry["refactor_state"] == "implementation-extracted" and not entry["module_owner"]
                   for entry in status["workflows"])


def test_implementation_wrappers_import_their_moved_modules():
    status = json.loads((ROOT / "docs/development/SCRIPT-STATUS.json").read_text(encoding="utf-8"))
    for row in status["workflows"]:
        if row["refactor_state"] != "implementation-extracted":
            continue
        expected = row["module_owner"].removesuffix(".py").replace("/", ".")
        source = (ROOT / row["legacy_path"]).read_text(encoding="utf-8")
        assert f"from {expected} import" in source
