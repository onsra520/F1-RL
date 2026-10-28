"""Actual owner-profile numerical witnesses, source-bound replay and Gym."""
from pathlib import Path
import pytest
from src.validation.wp7_executed_feasibility import certify_spatial_profile_execution,replay_planar_execution
ROOT=Path(__file__).resolve().parents[2]
PROFILE='spatial_rotating_owner_v3'


@pytest.mark.parametrize('track,seed,reset',[('straight-sprint-v1',0,None),('slalom-v1',0,0),('right-turn-v1',2,0),('ramp-v1',1,0)])
def test_declared_owner_selection_executes_ordered_finish_stop_and_actual_gym(track,seed,reset,monkeypatch,tmp_path):
    r=certify_spatial_profile_execution(ROOT,track,profile=PROFILE,seed_index=seed,reset_rank=0,reset_index=reset)
    assert r['status']=='executable_numerical_registered_spatial_path',r['blockers']
    assert r['schema_version']=='wp7_executed_spatial_profile_feasibility_v2'
    assert r['execution_profile']==PROFILE and r['finished'] and r['stopped']
    assert r['physics']['ground_motion_version']=='spatial_rotating_compliant_lifecycle_v3'
    assert r['spatial_work_audit']['max_energy_residual_fraction']<=1.
    assert not r['training_authorized'] and not r['scientific_evidence']
    if reset is None:
        assert replay_planar_execution(ROOT,r,expected_sha256=r['sha256'])==r
    from scripts.wp7_train_acceptance import verify_gym_finish_prefix
    prefix=verify_gym_finish_prefix(ROOT,r)
    assert prefix['status']=='successful_finish' and prefix['directed_crossings']==r['expected_gate_ids']
    if track=='ramp-v1':
        assert prefix['mixed_normal_loaded_frames']>0
    if track=='straight-sprint-v1':
        import src.training.env_factory as factory_module
        from src.rl.gym_environment import F1RLEnv
        close=F1RLEnv.close
        def foreign_after_close(env):
            close(env)
            monkeypatch.setattr(factory_module,'__file__',str(tmp_path/'foreign'/'env_factory.py'))
        monkeypatch.setattr(F1RLEnv,'close',foreign_after_close)
        with pytest.raises(ValueError,match='foreign F1RL source'):
            verify_gym_finish_prefix(ROOT,r)
