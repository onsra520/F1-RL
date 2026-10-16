"""Evidence-aware engineering traceability for F1-RL non-web core.

A documented implementation is not a tested pass. A software pass is not
empirical validation, independent review or training authorization.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from .wp7_protocol import digest

MATRIX_PATH = "configs/research/wp7-core-readiness-matrix-v1.json"


def load_readiness_matrix(root: Path) -> dict:
    document = json.loads((root / MATRIX_PATH).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema_version", "scope", "target_fraction", "subsystems"}:
        raise ValueError("readiness matrix requires exact root fields")
    if document["schema_version"] != "wp7_core_traceability_v1" or document["scope"] != "software_and_compatibility_only_no_scientific_claim":
        raise ValueError("unregistered readiness matrix")
    if type(document["target_fraction"]) is not float or document["target_fraction"] != 0.9:
        raise ValueError("readiness threshold is locked to 0.9")
    subsystems = document["subsystems"]
    if not isinstance(subsystems, list) or len(subsystems) != 9:
        raise ValueError("exactly nine non-web subsystems required")
    seen = set()
    for subsystem in subsystems:
        if not isinstance(subsystem, dict) or set(subsystem) != {"id", "capabilities", "compatibility"}:
            raise ValueError("invalid subsystem schema")
        name = subsystem["id"]
        if not isinstance(name, str) or name in seen or not name:
            raise ValueError("duplicate/empty subsystem")
        seen.add(name)
        cases = subsystem["capabilities"]
        if not isinstance(cases, list) or len(cases) < 2:
            raise ValueError("missing functional capabilities")
        keys = set()
        total = 0
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"id", "weight", "status", "priority", "source", "test", "blocker"}:
                raise ValueError("invalid capability schema")
            identity = case["id"]
            if not isinstance(identity, str) or not identity or identity in keys:
                raise ValueError("duplicate/empty capability")
            keys.add(identity)
            if type(case["weight"]) is not int or case["weight"] <= 0:
                raise ValueError("positive integer weight required")
            total += case["weight"]
            if case["priority"] not in ("P0", "P1") or case["status"] not in ("implemented", "partial", "blocked"):
                raise ValueError("invalid capability priority/status")
            if case["status"] != "implemented" and not isinstance(case["blocker"], str):
                raise ValueError("unresolved capability requires a blocker")
            for key in ("source", "test"):
                rel = Path(case[key])
                if (not isinstance(case[key], str) or rel.is_absolute()
                        or ".." in rel.parts or not (root / rel).is_file()
                        or not (root / rel).resolve().is_relative_to(root)):
                    raise ValueError("capability references missing/unsafe local evidence")
        if total != 100:
            raise ValueError("functional denominator must equal 100 per subsystem")
        comp = subsystem["compatibility"]
        if (not isinstance(comp, dict) or set(comp) != {"status", "required"}
                or comp["status"] != "in_progress"
                or not isinstance(comp["required"], list) or len(comp["required"]) != 3):
            raise ValueError("compatibility is a separate unfinished engineering gate")
    return document


def evaluate_readiness(repository_root: Path | str, *, verify: bool = False) -> dict:
    if type(verify) is not bool:
        raise ValueError("verify must be bool")
    root = Path(repository_root).resolve(strict=True)
    doc = load_readiness_matrix(root)
    tests = sorted({case["test"] for row in doc["subsystems"] for case in row["capabilities"] if case["status"] == "implemented"})
    receipt = None
    tests_verified = False
    if verify:
        env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTEST_ADDOPTS", None)
        command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *tests]
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=240)
        receipt = {
            "command": command, "returncode": completed.returncode,
            "stdout_sha256": sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": sha256(completed.stderr.encode()).hexdigest(),
            "tail": completed.stdout[-1200:], "stderr": completed.stderr[-1000:],
        }
        tests_verified = completed.returncode == 0
    subsystems = []
    for row in doc["subsystems"]:
        cases = []
        accepted_weight = 0
        unresolved_p0 = []
        for case in row["capabilities"]:
            covered = case["status"] == "implemented" and tests_verified
            if covered:
                accepted_weight += case["weight"]
            elif case["priority"] == "P0":
                unresolved_p0.append(case["id"])
            cases.append({
                **case, "verified": covered,
                "source_sha256": sha256((root / case["source"]).read_bytes()).hexdigest(),
                "test_sha256": sha256((root / case["test"]).read_bytes()).hexdigest(),
            })
        score = accepted_weight / 100.0
        eligible = score >= doc["target_fraction"] and not unresolved_p0
        subsystems.append({
            "id": row["id"], "functional_fraction_verified": score,
            "documented_implementation_fraction": sum(c["weight"] for c in row["capabilities"] if c["status"] == "implemented") / 100,
            "functional_gate": "PASS" if eligible else "INCOMPLETE",
            "unresolved_p0": unresolved_p0,
            "compatibility_gate": "IN_PROGRESS",
            "cases": cases,
        })
    report = {
        "schema_version": "wp7_core_readiness_report_v1",
        "scope": doc["scope"], "target_fraction": doc["target_fraction"],
        "matrix_sha256": sha256((root / MATRIX_PATH).read_bytes()).hexdigest(),
        "software_test_verification": receipt,
        "subsystems": subsystems, "all_subsystems_functional_90": all(s["functional_gate"] == "PASS" for s in subsystems),
        "compatibility_complete": False, "scientific_acceptance": "NOT_EVALUATED",
        "training_authorized": False,
    }
    return {**report, "sha256": digest(report)}
