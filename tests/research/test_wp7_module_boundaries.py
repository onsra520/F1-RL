"""WP7 keeps registration and scenario construction behind a stable facade."""

import ast
from pathlib import Path

from src.validation import wp7_protocol, wp7_registration, wp7_scenarios


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_facade_reexports_registration_and_scenario_entrypoints():
    assert wp7_protocol.PROFILES is wp7_registration.PROFILES
    assert wp7_protocol._unique is wp7_registration._unique
    assert wp7_protocol.load_registration is wp7_registration.load_registration
    assert wp7_protocol.protocol_profile is wp7_registration.protocol_profile
    assert wp7_protocol.prepare_scenarios is wp7_scenarios.prepare_scenarios


def test_wp7_registration_and_scenarios_do_not_import_protocol_facade():
    for module_name in ("wp7_registration.py", "wp7_scenarios.py"):
        tree = ast.parse(
            (ROOT / "src/validation" / module_name).read_text(encoding="utf-8")
        )
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert "wp7_protocol" not in imports
