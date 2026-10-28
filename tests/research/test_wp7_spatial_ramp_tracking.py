"""Registered Ramp reset tracking uses actual numerical physics and Gym."""
from pathlib import Path
import numpy as np
import pytest

from src.research.wp7_executed_feasibility import certify_spatial_profile_execution,replay_planar_execution
from src.research.wp7_protocol import load_registration,prepare_scenarios

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('seed_index',[0,1,2])
def test_spatial_ramp_corrects_declared_heading_and_lateral_reset_before_finish(seed_index):
    r=certify_spatial_profile_execution(ROOT,'ramp-v1',seed_index=seed_index,reset_index=0)
    assert r['status']=='executable_numerical_registered_spatial_path',r['blockers']
    assert r['finished'] and r['stopped'] and r['gate_ids']==r['expected_gate_ids']
    assert r['max_sampled_route_deviation_m']<4.
    assert any(abs(c[0])>1e-3 for c in r['controls'])
    assert r['controller']['version']=='spatial_ramp_route_speed_stop_v1'
    assert r['reset_metadata']['seed']==1000+seed_index
    assert not r['training_authorized'] and not r['scientific_evidence']
    if seed_index==1:
        assert replay_planar_execution(ROOT,r,expected_sha256=r['sha256'])==r
    f=next(s.factory for s in prepare_scenarios(ROOT,load_registration(ROOT,profile='spatial_ground_v1'),seed_index=seed_index).scenarios
           if s.track_id=='ramp-v1')
    env=f.create(reset_rank=0,reset_sequence=0)
    try:
        observation,_=env.reset(seed=144)
        for control in r['controls']:
            observation,_,terminated,truncated,info=env.step(np.asarray(control,dtype=np.float32))
            assert env.observation_space.contains(observation) and not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
    finally:
        env.close()
