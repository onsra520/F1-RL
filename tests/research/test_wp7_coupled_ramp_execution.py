"""Actual TRAIN asset, force-coupled engineering proposal and pinned replay."""
from pathlib import Path
from dataclasses import replace

import numpy as np

from src.validation.wp7_executed_feasibility import certify_coupled_ramp_execution, replay_planar_execution
from src.rl.action import ActionContract
from tests.integration.test_coupled_ground_motion import coupled_factory


def test_actual_ramp_executes_coupled_contact_forces_and_replays():
    root = Path(__file__).resolve().parents[2]
    record = certify_coupled_ramp_execution(root)
    assert record['status'] == 'executable_numerical_ramp_path',record['blockers']
    assert record['schema_version'] == 'wp7_executed_coupled_ramp_proposal_v1'
    assert record['profile_registration_status'] == 'engineering_proposal_not_training_registration'
    assert record['registered_physics_sha256'] != record['physics_sha256']
    assert record['physics']['longitudinal_model'] == 'coupled_contact_force_rk4_v1'
    assert record['gate_ids'] == record['full_body_gate_passages'] == ['checkpoint-0','checkpoint-1','finish']
    assert record['finished'] and record['stopped']
    assert record['coupled_work_audit']['accepted_substeps'] == record['proof_substeps']
    assert record['coupled_work_audit']['max_energy_residual_fraction']<=1.
    assert not record['training_authorized'] and not record['scientific_evidence']
    factory = replace(coupled_factory(),action_contract=ActionContract('steering-throttle-brake-v2'))
    assert factory.physics_config.sha256 == record['physics_sha256']
    env = factory.create()
    try:
        observation,_ = env.reset(seed=17)
        for control in record['controls']:
            observation,_,terminated,truncated,info = env.step(np.array(control,dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert info['physics']['coupled_ground']['version'] == 'coupled_longitudinal_rk4_work_v1'
            assert not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind'] == 'successful_finish'
    finally:
        env.close()
    assert replay_planar_execution(root,record,expected_sha256=record['sha256']) == record
