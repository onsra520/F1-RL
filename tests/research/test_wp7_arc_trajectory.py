"""Continuous geometry of real TRAIN paths, not learned behavior or admission."""
from dataclasses import replace
from math import pi
from pathlib import Path
import json

import pytest

from src.research.wp7_arc_trajectory import ArcPath, certify_arc_trajectory, search_right_turn_trajectory
from src.research.wp7_protocol import digest
from src.sandbox.geometry.vector import Vec3
from src.sandbox.geometry.rotation import Quaternion
from src.track_assets import load_track_asset

ROOT = Path(__file__).resolve().parents[2]


def asset(name='right-turn-v1'):
    return load_track_asset(ROOT/f'assets/tracks/training/{name}.json')


def test_actual_right_turn_has_continuously_certified_full_body_passage():
    track = asset()
    result = search_right_turn_trajectory(track)
    assert result['geometry_status'] == 'certified_continuous_planar_path'
    assert result['world_sha256'] == track.world_sha256
    assert [g['gate_id'] for g in result['gate_passages']] == ['checkpoint-0','checkpoint-1','finish']
    assert all(g['body_passage_certified'] for g in result['gate_passages'])
    assert result['gate_passages'][1]['crossing_s_m'] < result['gate_passages'][2]['crossing_s_m']
    brackets = [g['center_crossing_bracket_s_m'] for g in result['gate_passages']]
    assert all(lo<g['crossing_s_m']<hi for (lo,hi),g in zip(brackets,result['gate_passages']))
    assert all(a[1]<b[0] for a,b in zip(brackets,brackets[1:]))
    assert result['speed_contact_status'] == 'unresolved'
    assert result['training_authorized'] is False
    assert result['initial_spawn_xz_heading_aligned']
    assert result['route_corridor']['status'] == 'hard_violation'
    assert result['route_corridor']['witness']['deviation_m'] > 4.
    assert result['executable_feasibility'] is False
    assert 'hard_route_corridor_hard_violation' in result['execution_blockers']
    record = json.loads(json.dumps(result))
    path = ArcPath(tuple(record['path']['start_xz_m']),record['path']['start_yaw_rad'],
                   tuple(tuple(s) for s in record['path']['segments_length_curvature']))
    replay = json.loads(json.dumps(certify_arc_trajectory(track,path)))
    assert {k:v for k,v in record.items() if k not in ('sha256','search_attempts','search_scope')} == {
        k:v for k,v in replay.items() if k!='sha256'}
    assert replay['sha256'] in [a['certificate_sha256'] for a in record['search_attempts']]
    assert len(record['search_attempts']) == 3
    assert record['sha256'] == digest({k:v for k,v in record.items() if k != 'sha256'})


def test_straight_path_is_certified_but_partial_finish_is_not():
    track = asset('straight-sprint-v1')
    path = ArcPath((0.,4.),0.,((112.,0.),))
    assert certify_arc_trajectory(track,path)['geometry_status'] == 'certified_continuous_planar_path'
    partial = replace(path,segments_length_curvature=((108.1,0.),))
    result = certify_arc_trajectory(track,partial)
    assert result['geometry_status'] == 'unresolved'
    assert 'full_body_passage_unresolved:finish' in result['blockers']


def test_narrow_gate_rejects_center_only_success():
    track = asset('straight-sprint-v1')
    first,second = track.world.checkpoints
    first = replace(first,volume=replace(first.volume,half_extents=Vec3(.89,2.,1.)))
    track = replace(track,world=replace(track.world,checkpoints=(first,second)))
    result = certify_arc_trajectory(track,ArcPath((0.,4.),0.,((112.,0.),)))
    assert result['geometry_status'] == 'unresolved'
    assert 'full_body_passage_unresolved:checkpoint-0' in result['blockers']


def test_curvature_and_nonplanar_domain_fail_closed():
    path = ArcPath((0.,4.),0.,((1.,1.),))
    assert 'steering_curvature_bound_exceeded' in certify_arc_trajectory(asset(),path)['blockers']
    assert 'noncoplanar_or_sloped_support_unregistered' in certify_arc_trajectory(asset('ramp-v1'),path)['blockers']


def test_rotated_gate_uses_its_actual_normal_and_aperture():
    track = asset('straight-sprint-v1')
    orientation = Quaternion.from_axis_angle(Vec3(0.,1.,0.),pi/2)
    finish = replace(track.world.finish_lines[0],volume=replace(track.world.finish_lines[0].volume,
        center=Vec3(0.,0.,60.),orientation=orientation))
    track = replace(track,spawn=replace(track.spawn,position=Vec3(-3.,0.,60.),orientation=orientation),
                    world=replace(track.world,checkpoints=(),finish_lines=(finish,)))
    result = certify_arc_trajectory(track,ArcPath((-3.,60.),pi/2,((6.,0.),)))
    assert result['geometry_status'] == 'certified_continuous_planar_path'
    assert result['gate_passages'][0]['crossing_s_m'] == pytest.approx(3.)
    assert result['route_corridor']['status'] == 'clear'
    assert result['initial_spawn_xz_heading_aligned']


def test_reordered_gates_cannot_be_credited_backwards():
    track = asset('straight-sprint-v1')
    a,b = track.world.checkpoints
    track = replace(track,world=replace(track.world,checkpoints=(replace(a,order=1),replace(b,order=0))))
    result = certify_arc_trajectory(track,ArcPath((0.,4.),0.,((112.,0.),)))
    assert result['geometry_status'] == 'unresolved'
    assert 'full_body_passage_unresolved:checkpoint-0' in result['blockers']


def test_path_starting_with_body_in_gate_is_not_a_full_passage():
    result = certify_arc_trajectory(asset('straight-sprint-v1'),ArcPath((0.,34.),0.,((82.,0.),)))
    assert 'full_body_passage_unresolved:checkpoint-0' in result['blockers']
    assert result['initial_spawn_xz_heading_aligned'] is False


def test_thin_off_center_obstacle_blocks_the_continuous_body():
    from src.sandbox.entities.obstacle import Obstacle
    from src.sandbox.track.volume import OrientedBoxVolume
    track = asset('straight-sprint-v1')
    obstacle = Obstacle('thin',OrientedBoxVolume(Vec3(.8,.8,20.123),Vec3(.05,.1,.001),Quaternion.identity()))
    track = replace(track,world=replace(track.world,obstacles=(obstacle,)))
    result = certify_arc_trajectory(track,ArcPath((0.,4.),0.,((112.,0.),)))
    assert result['geometry_status'] == 'unresolved'
    assert 'continuous_body_road_or_obstacle_unresolved' in result['blockers']


@pytest.mark.parametrize('segments', [((0.,0.),), ((float('nan'),0.),), ((10.,True),), ((1001.,0.),)])
def test_invalid_paths_rejected(segments):
    with pytest.raises(ValueError):
        ArcPath((0.,4.),0.,segments)


def test_underflowed_derived_curvature_is_an_explicit_numeric_domain_error():
    from src.research.wp7_track_feasibility import PARAMETERS
    with pytest.raises(ValueError,match='numeric'):
        search_right_turn_trajectory(asset(),parameters=replace(PARAMETERS,
            wheelbase=1e308,max_steering_angle=1e-300))


def test_tangent_touch_is_not_a_directed_passage():
    track = asset('straight-sprint-v1')
    road = replace(track.world.surfaces[0],width=60.)
    finish = replace(track.world.finish_lines[0],volume=replace(track.world.finish_lines[0].volume,
        center=Vec3(10.,0.,14.)))
    track = replace(track,world=replace(track.world,surfaces=(road,),checkpoints=(),finish_lines=(finish,)))
    result = certify_arc_trajectory(track,ArcPath((0.,4.),0.,((pi/.1,.1),)))
    assert result['geometry_status'] == 'unresolved'
    assert 'full_body_passage_unresolved:finish' in result['blockers']


def test_real_train_auditor_dispatches_arc_search_and_preserves_default_report():
    from src.research.wp7_track_feasibility import audit_train_tracks
    legacy = audit_train_tracks(ROOT)
    assert all('alternative_trajectory' not in row for row in legacy['tracks'])
    report = audit_train_tracks(ROOT,search_alternatives=True)
    row = next(r for r in report['tracks'] if r['track_id']=='right-turn-v1')
    assert row['alternative_trajectory']['schema_version'] == 'wp7_arc_path_certificate_v1'
    assert row['alternative_trajectory']['geometry_status'] == 'certified_continuous_planar_path'
    assert row['alternative_trajectory']['executable_feasibility'] is False
    assert report['training_authorized'] is False
