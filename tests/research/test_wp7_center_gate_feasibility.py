"""Owner 5A center gates are distinct from optional whole-body apertures."""
from pathlib import Path
from dataclasses import replace

import pytest

from src.research.wp7_arc_trajectory import certify_arc_trajectory, search_right_turn_trajectory, ArcPath
from src.track_assets import load_track_asset

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def registered_audit():
    from src.research.wp7_track_feasibility import audit_train_tracks
    return audit_train_tracks(ROOT,search_alternatives=True,execute_controls=True,registered_center_gates=True)


def test_actual_right_turn_has_continuous_body_road_and_owner_center_gate_path():
    asset = load_track_asset(ROOT/'assets/tracks/training/right-turn-v1.json')
    result = search_right_turn_trajectory(asset,gate_contract='directed_center_crossing_v1')
    assert result['schema_version'] == 'wp7_arc_path_certificate_v2'
    assert result['gate_contract'] == 'directed_center_crossing_v1'
    assert result['geometry_status'] == 'certified_continuous_planar_path',result['blockers']
    assert result['route_corridor']['status'] == 'clear'
    assert result['route_corridor']['deviation_upper_bound_m'] < 4.
    assert [p['gate_id'] for p in result['gate_passages']] == ['checkpoint-0','checkpoint-1','finish']
    assert all(p['center_crossing_certified'] for p in result['gate_passages'])
    assert not any(p.get('body_passage_certified',False) for p in result['gate_passages'])
    path = result['path']
    program = ArcPath(tuple(path['start_xz_m']),path['start_yaw_rad'],
                      tuple(tuple(s) for s in path['segments_length_curvature']))
    strict = certify_arc_trajectory(asset,program)
    assert strict['schema_version'] == 'wp7_arc_path_certificate_v1'
    assert strict['geometry_status'] == 'unresolved'
    assert 'full_body_passage_unresolved:checkpoint-1' in strict['blockers']
    assert not result['executable_feasibility'] and not result['training_authorized']


def test_actual_registered_right_turn_executes_center_gates_road_braking_and_replay(registered_audit):
    import numpy as np
    from src.learning.action import ActionContract
    from src.research.wp7_protocol import load_registration, prepare_scenarios
    from src.research.wp7_executed_feasibility import certify_registered_center_execution, replay_planar_execution
    result = next(row['executed_planar_trajectory'] for row in registered_audit['tracks'] if row['track_id']=='right-turn-v1')
    assert result['status'] == 'executable_numerical_registered_center_path',result['blockers']
    assert result['gate_contract'] == 'directed_center_crossing_v1'
    assert result['physics']['ground_motion_version'] == 'arc_distance_v1'
    assert result['gate_ids'] == result['expected_gate_ids'] == ['checkpoint-0','checkpoint-1','finish']
    assert result['finished'] and result['stopped']
    assert result['max_sampled_route_deviation_m']<4.
    assert result['proof_substeps']>=16*result['step_count']
    assert 'full_body_gate_passages' not in result
    assert not result['training_authorized'] and not result['scientific_evidence']
    factory = next(s.factory for s in prepare_scenarios(ROOT,load_registration(ROOT),seed_index=0).scenarios
                   if s.track_id=='right-turn-v1')
    assert result['physics_sha256'] == factory.physics_config.sha256
    factory = replace(factory,reset_config=None,reset_track_id=None,
                      action_contract=ActionContract('steering-throttle-brake-v2'))
    env = factory.create()
    try:
        observation,_ = env.reset(seed=19)
        for control in result['controls']:
            observation,_,terminated,truncated,info = env.step(np.array(control,dtype=np.float32))
            assert env.observation_space.contains(observation)
            assert not truncated
            if terminated:
                break
        assert terminated and info['research_outcome']['kind']=='successful_finish'
        assert env.requires_scientific_admission
    finally:
        env.close()
    assert replay_planar_execution(ROOT,result,expected_sha256=result['sha256']) == result


def test_all_train_center_audit_keeps_physics_assets_and_admission_identities(registered_audit):
    from src.research.wp7_track_feasibility import audit_train_tracks
    old = audit_train_tracks(ROOT)
    assert old['schema_version']=='wp7_body_track_audit_v1' and 'gate_contract' not in old
    assert registered_audit['schema_version']=='wp7_body_track_audit_v2'
    assert registered_audit['gate_contract']=='directed_center_crossing_v1'
    assert registered_audit['manifest_sha256']==old['manifest_sha256']
    assert registered_audit['training_authorized'] is False
    expected = {'straight-sprint-v1','slalom-v1','right-turn-v1','ramp-v1'}
    assert {r['track_id'] for r in registered_audit['tracks']}==expected
    for row in registered_audit['tracks']:
        result = row['executed_planar_trajectory']
        assert result['schema_version']=='wp7_executed_registered_center_feasibility_v1'
        assert result['gate_contract']=='directed_center_crossing_v1'
        assert result['owner_contract_sha256']==registered_audit['owner_contract_sha256']
        assert result['training_authorized'] is False and result['scientific_evidence'] is False
        assert result['world_sha256']==next(r['world_sha256'] for r in old['tracks'] if r['track_id']==row['track_id'])
        if row['track_id']=='ramp-v1':
            assert result['status']=='unresolved' and result['blockers']
        else:
            assert result['status']=='executable_numerical_registered_center_path',result['blockers']
            assert result['finished'] and result['stopped']


@pytest.mark.parametrize('change',['gate_miss','reordered','narrow_road','body_obstacle'])
def test_center_gate_mode_preserves_finite_gate_order_and_full_body_geometry(change):
    from src.sandbox.geometry.vector import Vec3
    from src.sandbox.geometry.rotation import Quaternion
    from src.sandbox.entities.obstacle import Obstacle
    from src.sandbox.track.volume import OrientedBoxVolume
    asset=load_track_asset(ROOT/'assets/tracks/training/right-turn-v1.json')
    reference=search_right_turn_trajectory(asset,gate_contract='directed_center_crossing_v1')['path']
    path=ArcPath(tuple(reference['start_xz_m']),reference['start_yaw_rad'],
                 tuple(tuple(s) for s in reference['segments_length_curvature']))
    world=asset.world
    first,second=world.checkpoints
    if change=='gate_miss':
        second=replace(second,volume=replace(second.volume,center=Vec3(22.,0.,70.)))
        world=replace(world,checkpoints=(first,second))
    elif change=='reordered':
        world=replace(world,checkpoints=(replace(first,order=1),replace(second,order=0)))
    elif change=='narrow_road':
        world=replace(world,surfaces=(world.surfaces[0],replace(world.surfaces[1],width=1.7)))
    else:
        # Centreline is clear; the obstacle intersects the yawed body's side.
        obstacle=Obstacle('body-side',OrientedBoxVolume(Vec3(40.,.8,70.53),Vec3(.01,.1,.01),Quaternion.identity()))
        world=replace(world,obstacles=(obstacle,))
    result=certify_arc_trajectory(replace(asset,world=world),path,gate_contract='directed_center_crossing_v1')
    assert result['geometry_status']=='unresolved' and result['blockers']
    assert not result['executable_feasibility'] and not result['training_authorized']


@pytest.mark.parametrize('contract',[None,True,'point_event_v1'])
def test_unregistered_gate_semantics_fail_closed(contract):
    asset=load_track_asset(ROOT/'assets/tracks/training/right-turn-v1.json')
    with pytest.raises(ValueError,match='gate passage contract'):
        search_right_turn_trajectory(asset,gate_contract=contract)


def test_center_execution_rejects_foreign_track_before_source_access(tmp_path):
    from src.research.wp7_executed_feasibility import certify_registered_center_execution
    with pytest.raises(ValueError,match='TRAIN'):
        certify_registered_center_execution(tmp_path/'missing','not-TRAIN')


def test_center_replay_rejects_self_rehashed_gate_claim(registered_audit):
    from copy import deepcopy
    from src.research.wp7_executed_feasibility import replay_planar_execution
    from src.research.wp7_protocol import digest
    result=deepcopy(next(row['executed_planar_trajectory'] for row in registered_audit['tracks']
                         if row['track_id']=='right-turn-v1'))
    result['whole_body_gate_aperture_required']=True
    result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'})
    with pytest.raises(ValueError,match='physical evidence identity differs'):
        replay_planar_execution(ROOT,result,expected_sha256=result['sha256'])


@pytest.mark.parametrize('selected',[None,1,'directed_center_crossing_v1',True])
def test_center_audit_rejects_ambiguous_or_unused_selection(selected):
    from src.research.wp7_track_feasibility import audit_train_tracks
    with pytest.raises(ValueError,match='registered center gates'):
        audit_train_tracks(ROOT,registered_center_gates=selected)
