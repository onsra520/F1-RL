"""Fixed 12-case scripted TRAIN acceptance; no PPO learning or scientific permission."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from f1rl.learning.pilot_preflight import TRAIN_IDS
from f1rl.research.wp7_executed_feasibility import (
    certify_spatial_profile_execution,replay_planar_execution,_execution_sources,_verify_module_origins,
)
from f1rl.research.wp7_protocol import build_protocol,digest,load_registration,prepare_scenarios
from f1rl.track_assets.canonical import fingerprint_track_world

PROFILE='spatial_rotating_owner_v3'


def verify_gym_finish_prefix(root,certificate):
    """Check a current-source certificate's float32 Gym finish prefix, not its stop suffix."""
    c=certificate
    if (not isinstance(c,dict) or c.get('sha256')!=digest({k:v for k,v in c.items() if k!='sha256'})
            or c.get('schema_version')!='wp7_executed_spatial_profile_feasibility_v2'
            or c.get('execution_profile')!=PROFILE or c.get('track_id') not in TRAIN_IDS
            or c.get('status')!='executable_numerical_registered_spatial_path'
            or c.get('finished') is not True or c.get('stopped') is not True or c.get('blockers')!=[]
            or c.get('training_authorized') is not False or c.get('scientific_evidence') is not False
            or type(c.get('paired_seed_index')) is not int or c['paired_seed_index'] not in range(3)
            or type(c.get('reset_rank')) is not int or c['reset_rank']!=0
            or not (c.get('reset_index') is None or type(c['reset_index']) is int and c['reset_index']==0)):
        raise ValueError('current bounded TRAIN execution certificate required')
    _verify_module_origins(root)
    protocol=build_protocol(root,profile=PROFILE)
    if c['protocol_sha256']!=protocol['sha256']:
        raise ValueError('certificate protocol/source differs')
    f=next(row.factory for row in prepare_scenarios(root,load_registration(root,profile=PROFILE),
        seed_index=c['paired_seed_index']).scenarios if row.track_id==c['track_id'])
    if f.physics_config.sha256!=c['physics_sha256'] or fingerprint_track_world(f.track)!=c['world_sha256']:
        raise ValueError('certificate physics/world differs')
    if c['reset_index'] is None:
        f=replace(f,reset_config=None,reset_track_id=None)
    env=f.create(reset_rank=0,reset_sequence=c['reset_index'] or 0)
    actions=sha256();observations=sha256();crossings=[];mixed_frames=0;leaves=0
    max_speed=0.;max_energy_fraction=0.;normals={p.id:p.normal for p in f.track.surfaces}
    try:
        observation,info=env.reset()
        if not env.observation_space.contains(observation):
            raise ValueError('invalid reset observation')
        if c['reset_index'] is not None and json.loads(json.dumps(info['train_reset']))!=c['reset_metadata']:
            raise ValueError('reset metadata differs')
        observations.update(observation.tobytes())
        terminated=False
        for index,action in enumerate(c['controls']):
            action=np.asarray(action,dtype=np.float32)
            if not env.action_space.contains(action):
                raise ValueError('control outside registered action contract')
            observation,reward,terminated,truncated,info=env.step(action)
            if (not env.observation_space.contains(observation) or not np.isfinite(reward) or truncated
                    or info['physics']['sha256']!=c['physics_sha256']):
                raise ValueError('invalid Gym transition or physics identity')
            actions.update(action.tobytes());observations.update(observation.tobytes())
            crossings.extend(info['race']['directed_crossings'])
            max_speed=max(max_speed,info['telemetry']['vehicle_speed'])
            if max_speed>f.vehicle_parameters.max_ground_speed:
                raise ValueError('registered speed bound exceeded')
            for leaf in info['physics']['spatial_ground']['substeps']:
                if (abs(leaf['energy_residual_j'])>leaf['energy_tolerance_j'] or leaf['dissipation_j']<0.
                        or leaf['linear_impulse_residual_ns']>1e-7 or leaf['angular_impulse_residual_nms']>1e-7):
                    raise ValueError('invalid work or momentum audit')
                leaves+=1
                max_energy_fraction=max(max_energy_fraction,abs(leaf['energy_residual_j'])/leaf['energy_tolerance_j'])
                for stage in leaf['contact_stages']:
                    mixed_frames+=len({normals[c['patch_id']] for c in stage if c['normal_force_n']>0.})>1
            if terminated:
                break
        if (not terminated or info['research_outcome']['kind']!='successful_finish'
                or info['research_outcome']['valid_sample'] is not True or crossings!=c['expected_gate_ids']):
            raise ValueError('ordered valid Gym finish required')
        if build_protocol(root,profile=PROFILE)['sha256']!=protocol['sha256']:
            raise ValueError('source changed during Gym verification')
        result=dict(schema_version='wp7_train_gym_finish_prefix_v1',status='successful_finish',
            certificate_sha256=c['sha256'],protocol_sha256=protocol['sha256'],steps=index+1,
            reset_metadata=c['reset_metadata'],directed_crossings=crossings,
            max_sampled_speed_mps=max_speed,accepted_leaf_count=leaves,
            max_energy_residual_fraction=max_energy_fraction,mixed_normal_loaded_frames=mixed_frames,
            actions_float32_sha256=actions.hexdigest(),observations_sha256=observations.hexdigest(),
            outcome=info['research_outcome'],training_authorized=False,scientific_evidence=False)
    finally:
        env.close()
    _verify_module_origins(root)
    return result


def _write_new(path,document):
    body={k:v for k,v in document.items() if k!='sha256'}
    identity=digest(body)
    if 'sha256' in document and document['sha256']!=identity:
        raise ValueError('evidence checksum differs')
    sealed={**body,'sha256':identity}
    with path.open('x',encoding='utf-8') as stream:
        json.dump(sealed,stream,indent=2,allow_nan=False)
    return sealed


def run_acceptance(root,output):
    root=Path(root).resolve(strict=True);output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('new evidence directory outside source tree required')
    _verify_module_origins(root)
    protocol=build_protocol(root,profile=PROFILE)
    if protocol['source']['dirty']:
        raise ValueError('clean source required before evidence allocation')
    sources=_execution_sources(root)
    output.mkdir(parents=True,exist_ok=False)
    rows=[];unchanged=True
    for track in TRAIN_IDS:
        for seed in range(3):
            row=dict(track_id=track,paired_seed_index=seed,reset_rank=0,reset_index=0)
            stem=f'{track}-seed{seed}'
            try:
                certificate=certify_spatial_profile_execution(root,track,profile=PROFILE,
                    seed_index=seed,reset_rank=0,reset_index=0)
                _write_new(output/(stem+'-certificate.json'),certificate)
                if (certificate['status']!='executable_numerical_registered_spatial_path'
                        or replay_planar_execution(root,certificate,expected_sha256=certificate['sha256'])!=certificate):
                    raise ValueError('executable finish/stop and full-record replay required')
                gym=_write_new(output/(stem+'-gym.json'),verify_gym_finish_prefix(root,certificate))
                row.update(status='passed',certificate_sha256=certificate['sha256'],
                    full_record_replay_equal=True,gym_sha256=gym['sha256'])
            except Exception as exc:
                row.update(status='failed_without_redraw',error_type=type(exc).__name__,error=str(exc))
            rows.append(_write_new(output/(stem+'-result.json'),row))
            unchanged=(_execution_sources(root)==sources
                and build_protocol(root,profile=PROFILE)['sha256']==protocol['sha256'])
            if not unchanged:
                break
        if not unchanged:
            break
    unchanged=(_execution_sources(root)==sources
        and build_protocol(root,profile=PROFILE)['sha256']==protocol['sha256'])
    _verify_module_origins(root)
    return _write_new(output/'summary.json',dict(schema_version='wp7_train_execution_acceptance_v1',
        scope='fixed_four_TRAIN_tracks_three_paired_seeds_reset0_scripted_numerical_acceptance_not_PPO_robustness',
        source=protocol['source'],source_files_sha256=digest(sources),protocol_sha256=protocol['sha256'],
        execution_profile=PROFILE,rows=rows,complete=len(rows)==12,source_unchanged=unchanged,
        all_pass=len(rows)==12 and unchanged and all(r['status']=='passed' for r in rows),
        permissions=dict(training_permitted=False,heldout_permitted=False,scientific_freeze_permitted=False,wp8_permitted=False),
        training_authorized=False,scientific_evidence=False,scientific_acceptance=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='new external evidence directory; never overwritten')
    args=parser.parse_args()
    report=run_acceptance(Path(__file__).resolve().parents[1],args.output)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report['all_pass'] else 2


if __name__=='__main__':
    raise SystemExit(main())
