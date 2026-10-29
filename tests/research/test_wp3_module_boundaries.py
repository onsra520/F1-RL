"""The campaign and evaluation modules have a one-way dependency only."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_does_not_import_campaign_private_helpers():
    path = ROOT / "src/interfaces/evaluation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(node.module == "campaign" or (node.module or "").endswith(".campaign")
                   for node in imports)


def test_campaign_and_evaluation_import_in_either_order_without_cycle():
    importlib.import_module("src.interfaces.evaluation")
    importlib.import_module("src.interfaces.campaign")
    importlib.import_module("src.interfaces.campaign")
    importlib.import_module("src.interfaces.evaluation")
