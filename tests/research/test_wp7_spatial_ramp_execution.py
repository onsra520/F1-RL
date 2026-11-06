"""Full actual TRAIN Ramp through the spatial force proposal and pinned replay."""
from pathlib import Path

import numpy as np

from f1rl.research.wp7_executed_feasibility import certify_spatial_ramp_execution, replay_planar_execution
from tests.integration.test_spatial_ground_motion import spatial_factory


def test_actual_spatial_ramp_executes_contact_gates_stop_gym_and_replays():
    root=Path(__file__).resolve().parents[2]
    r=certify_spatial_ramp_execution(root)
    assert r['schema_version']=='wp7_executed_spatial_ramp_proposal_v1'
    assert r['status']=='executable_numerical_ramp_path',r['blockers']
    assert r['registered_physics_sha256']!=r['physics_sha256']
    assert r['physics']['ground_motion_version']=='spatial_contact_v1'
    assert r['gate_ids']==r['expected_gate_ids']==r['full_body_gate_passages']
    assert r['finished'] and r['stopped'] and r['step_count']<=500
    assert r['spatial_work_audit']['accepted_substeps']==r['proof_substeps']
    assert r['spatial_work_audit']['max_energy_residual_fraction']<=1.
    assert not r['training_authorized'] and not r['scientific_evidence']
    env=spatial_factory().create()
    try:
        observation,_=env.reset(seed=23)
        for action in r['controls']:
            observation,_,terminated,truncated,info=env.step(np.array(action,dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
        assert info['physics']['sha256']==r['physics_sha256']
        assert env.requires_scientific_admission
    finally:
        env.close()
    assert replay_planar_execution(root,r,expected_sha256=r['sha256'])==r
