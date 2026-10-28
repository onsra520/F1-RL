"""Actual TRAIN ramp execution in an explicit engineering physics proposal."""
from pathlib import Path
import pytest

from src.validation.wp7_executed_feasibility import certify_ramp_execution, replay_planar_execution


def test_actual_ramp_profile_executes_gates_contact_and_stop_and_replays():
    root = Path(__file__).resolve().parents[2]
    record = certify_ramp_execution(root)
    assert record['status'] == 'executable_numerical_ramp_path', record['blockers']
    assert record['profile_registration_status'] == 'engineering_proposal_not_training_registration'
    assert record['registered_physics_sha256'] != record['physics_sha256']
    assert record['physics']['body_clearance_version'] == 'bounded_patch_planes_v2'
    assert record['step_count'] <= 500
    assert record['finished'] and record['stopped']
    assert record['gate_ids'] == record['expected_gate_ids'] == record['full_body_gate_passages']
    assert record['gate_ids'] == ['checkpoint-0', 'checkpoint-1', 'finish']
    assert record['proof_substeps'] >= 16*record['step_count']
    assert all(c[0] == 0. for c in record['controls'])
    assert not record['training_authorized'] and not record['scientific_evidence']
    assert replay_planar_execution(root, record, expected_sha256=record['sha256']) == record


def test_ramp_tape_cannot_invent_steering_before_source_access(tmp_path):
    with pytest.raises(ValueError, match='zero steering'):
        certify_ramp_execution(tmp_path/'missing', controls=[[.1, .2, 0.]])
