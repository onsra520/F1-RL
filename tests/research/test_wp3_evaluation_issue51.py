"""Issue #51: the WP3 evaluator must pass bound report inputs and not overwrite outputs."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from f1rl.local_demo.evaluation import _write_new_or_match


ROOT = Path(__file__).resolve().parents[2]


def test_run_evaluation_reporter_receives_bound_model_and_cell_records():
    """Static regression: prevent the NameError after persisting the evaluation JSON."""
    tree = ast.parse((ROOT / "f1rl/local_demo/evaluation.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "run_evaluation")
    writes = [node for node in ast.walk(fn)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name)
              and node.func.id == "_write_evaluation_reports"]
    assert len(writes) == 1
    call = writes[0]
    assert [ast.unparse(arg) for arg in call.args] == [
        "campaign_root", "result", "model_records", "all_cells", "evaluation",
    ]
    assignments = {
        target.id
        for node in ast.walk(fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert {"model_records", "all_cells", "result", "evaluation"} <= assignments


def test_report_writer_restart_is_idempotent_and_never_overwrites(tmp_path):
    output = tmp_path / "WP3-CANDIDATE-COMPARISON.md"
    _write_new_or_match(output, "# Eval results\n")
    _write_new_or_match(output, "# Eval results\n")
    assert output.read_text(encoding="utf-8") == "# Eval results\n"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _write_new_or_match(output, "# Different evidence\n")
    assert output.read_text(encoding="utf-8") == "# Eval results\n"
