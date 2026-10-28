"""Traceability matrix is conservative: no green test alone grants scientific acceptance."""
import json
from pathlib import Path
import pytest

from src.research.wp7_readiness import load_readiness_matrix, evaluate_readiness, MATRIX_PATH

ROOT = Path(__file__).resolve().parents[2]


def test_nine_core_subsystems_have_explicit_denominators_and_unresolved_gates():
    matrix = load_readiness_matrix(ROOT)
    assert len(matrix["subsystems"]) == 9
    for row in matrix["subsystems"]:
        assert sum(c["weight"] for c in row["capabilities"]) == 100
        assert row["compatibility"]["status"] == "in_progress"
    report = evaluate_readiness(ROOT)
    assert report["all_subsystems_functional_90"] is False
    assert report["compatibility_complete"] is False
    assert report["scientific_acceptance"] == "NOT_EVALUATED"
    assert report["training_authorized"] is False
    assert all(row["functional_gate"] == "INCOMPLETE" for row in report["subsystems"])
    assert all(row["functional_fraction_verified"] == 0 for row in report["subsystems"])


@pytest.mark.parametrize("mutation", ("duplicate_capability", "missing_evidence", "bad_weight", "fake_pass"))
def test_matrix_denies_unregistered_changes(monkeypatch, mutation):
    document = load_readiness_matrix(ROOT)
    first = document["subsystems"][0]["capabilities"]
    if mutation == "duplicate_capability":
        first[1]["id"] = first[0]["id"]
    elif mutation == "missing_evidence":
        first[0]["test"] = "tests/does-not-exist.py"
    elif mutation == "bad_weight":
        first[0]["weight"] = 99
    else:
        first[0]["status"] = "PASS"
    original = Path.read_text
    def patched(self, *args, **kwargs):
        if self == ROOT / MATRIX_PATH:
            return json.dumps(document)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", patched)
    with pytest.raises(ValueError):
        load_readiness_matrix(ROOT)


def test_coverage_does_not_treat_documented_code_as_verified():
    report = evaluate_readiness(ROOT, verify=False)
    runtime = next(row for row in report["subsystems"] if row["id"] == "simulation_runtime")
    assert runtime["documented_implementation_fraction"] == 1.0
    assert runtime["functional_fraction_verified"] == 0.0
    assert runtime["functional_gate"] == "INCOMPLETE"
