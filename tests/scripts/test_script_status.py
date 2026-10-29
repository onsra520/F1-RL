"""Keep the grouped entrypoint inventory aligned with the workflow registry."""
import json
from pathlib import Path

from scripts.workflows.catalog import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]


def test_entrypoint_inventory_covers_every_grouped_workflow():
    status = json.loads((ROOT / "docs/development/SCRIPT-STATUS.json").read_text(encoding="utf-8"))
    records = {entry["entrypoint"]: entry for entry in status["workflows"]}
    assert len(records) == len(status["workflows"]) == len(WORKFLOWS) == 46
    for entry in WORKFLOWS:
        record = records[entry.script]
        assert (record["group"], record["alias"], record["scope"]) == (
            entry.group, entry.name, entry.scope
        )
        assert (ROOT / entry.script).is_file()
        assert (ROOT / record["former_root_path"]).is_file() is False
    assert status["counts"] == {
        "catalogued": 46,
        "grouped_entrypoints": 46,
        "root_legacy_entrypoints": 0,
        "compatibility_wrappers": 0,
        "retired_legacy_demo_entrypoints": 2,
    }
    assert {item["alias"] for item in status["retired_legacy_demo_entrypoints"]} == {
        "package-handoff",
        "launch-exploration",
    }
