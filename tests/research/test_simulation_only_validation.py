from pathlib import Path

from f1rl.research.simulation_validation import (
    load_simulation_validation_protocol,
    run_simulation_validation,
)

ROOT = Path(__file__).resolve().parents[2]


def test_versioned_simulation_validation_runs_repeatable_real_train_transitions():
    protocol, digest = load_simulation_validation_protocol(ROOT)
    result = run_simulation_validation(ROOT)

    assert protocol["schema_version"] == "f1rl_simulation_validation_v1"
    assert len(digest) == 64
    assert result["status"] == "SIMULATION_CONSISTENCY_PASS"
    assert result["physical_calibration_claim"] is False
    assert result["scientific_acceptance_claim"] is False
    assert result["registered_4m_boundary_preserved"] is True
    assert set(result["track_results"]) == set(protocol["registered_tracks"])
    assert all(row["status"] == "PASS" and row["steps"] == 4
               for row in result["track_results"].values())
