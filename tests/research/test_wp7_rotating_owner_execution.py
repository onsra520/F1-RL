"""Actual owner-profile numerical witnesses, source-bound replay and Gym."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from f1rl.research.wp7_executed_feasibility import certify_spatial_profile_execution,replay_planar_execution
from f1rl.research.wp7_protocol import load_registration,prepare_scenarios
ROOT=Path(__file__).resolve().parents[2]
PROFILE='spatial_rotating_owner_v3'


@pytest.mark.parametrize('track,seed,reset',[('straight-sprint-v1',0,None),('ramp-v1',1,0)])
def test_declared_owner_selection_executes_ordered_finish_stop_and_actual_gym(track,seed,reset):
    r=certify_spatial_profile_execution(ROOT,track,profile=PROFILE,seed_index=seed,reset_rank=0,reset_index=reset)
    assert r['status']=='executable_numerical_registered_spatial_path',r['blockers']
    assert r['schema_version']=='wp7_executed_spatial_profile_feasibility_v2'
    assert r['execution_profile']==PROFILE and r['finished'] and r['stopped']
    assert r['physics']['ground_motion_version']=='spatial_rotating_compliant_lifecycle_v3'
    assert r['spatial_work_audit']['max_energy_residual_fraction']<=1.
    assert not r['training_authorized'] and not r['scientific_evidence']
    if reset is None:
        assert replay_planar_execution(ROOT,r,expected_sha256=r['sha256'])==r
    f=next(row.factory for row in prepare_scenarios(ROOT,load_registration(ROOT,profile=PROFILE),seed_index=seed).scenarios
           if row.track_id==track)
    if reset is None:
        f=replace(f,reset_config=None,reset_track_id=None)
    normals={patch.id:patch.normal for patch in f.track.surfaces}
    mixed_normal_loaded_frames=0
    env=f.create(reset_rank=0,reset_sequence=reset or 0)
    try:
        observation,info=env.reset()
        if reset is not None:
            assert json.loads(json.dumps(info['train_reset']))==r['reset_metadata']
        for action in r['controls']:
            observation,_,terminated,truncated,info=env.step(np.asarray(action,dtype=np.float32))
            assert env.observation_space.contains(observation) and not truncated
            for leaf in info['physics']['spatial_ground']['substeps']:
                for frame in leaf['contact_stages']:
                    loaded={normals[c['patch_id']] for c in frame if c['normal_force_n']>0.}
                    mixed_normal_loaded_frames+=len(loaded)>1
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
        if track=='ramp-v1':
            assert mixed_normal_loaded_frames>0
    finally:
        env.close()
