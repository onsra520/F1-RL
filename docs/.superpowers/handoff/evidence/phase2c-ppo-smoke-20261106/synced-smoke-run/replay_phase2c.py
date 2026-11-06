from __future__ import annotations
import dataclasses, hashlib, json, math, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r'D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit').resolve()
EVIDENCE = Path(r'D:\Projects\.evidence\F1-RL\phase2c-ppo-smoke-20261106\synced-smoke-run')
PHASE1 = Path(r'D:\Projects\.evidence\F1-RL\phase1-physics-v4-audit-20261106')
PHASE2B = Path(r'D:\Projects\.evidence\F1-RL\phase2b-smoke-profile-wiring-20261106')
RESET_SEED = 20261106
STEP_LIMIT = 2500
SELECTED = (('C-seed0', 0), ('B-seed1', 1), ('A-seed2', 2))
TRACK_IDS = ('straight-sprint-v1', 'slalom-v1', 'right-turn-v1', 'ramp-v1')
EXPECTED_ASSETS = {
    'straight-sprint-v1': '9efe209780cfa8358f8622d38bb5d56db30425a57dcebebc16e6e447a10b5f00',
    'slalom-v1': 'e9dd588740189f7ed857de8e8b8f5eeb99f3161fdb9b8d3fa05d18978cdefa53',
    'right-turn-v1': 'a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e',
    'ramp-v1': '6c0da56966962363ed601c8fda2343006c207ee9ea6497deb72e6f6018721bd3',
}
EXPECTED_MODELS = {
    'C-seed0': '15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492',
    'B-seed1': '695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021',
    'A-seed2': 'e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc',
}
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info
from f1rl.research.wp3_eval_only_coordinator import (
    FULL06_RUNTIME_PROFILE, apply_runtime_profile, effective_runtime_profile,
)
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def safe(value):
    if dataclasses.is_dataclass(value):
        return safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        n = float(value)
        return n if math.isfinite(n) else str(n)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, '__dict__'):
        return safe(vars(value))
    return str(value)

def emit(stream, row):
    stream.write(json.dumps(safe(row), ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()

def wheel_support(info):
    physics = info.get('physics', {})
    telemetry = info.get('telemetry', {})
    spatial = physics.get('spatial_ground')
    last_stage = None
    counts = []
    if isinstance(spatial, dict):
        for substep in spatial.get('substeps', []):
            for stage in substep.get('contact_stages', []):
                counts.append(len(stage))
                last_stage = stage
    contacts = [] if last_stage is None else [
        {'wheel_index': x['wheel_index'], 'patch_id': x['patch_id'], 'normal_force_n': x['normal_force_n']}
        for x in last_stage
    ]
    return {
        'grounded': telemetry.get('grounded'),
        'spatial_supported_wheel_count_last_stage': None if last_stage is None else len(contacts),
        'spatial_supported_wheel_count_max_this_tick': max(counts) if counts else None,
        'last_stage_contacts': contacts,
    }

def terminal_projection(info):
    race = info.get('race', {})
    outcome = info.get('research_outcome', {})
    return {
        'finished': info.get('finished', False),
        'finished_now': info.get('finished_now', False),
        'terminal_reason': race.get('terminal_reason'),
        'terminal_failure': race.get('terminal_failure'),
        'hard_route_violation': race.get('terminal_reason') == 'hard_route_violation'
            or outcome.get('reason') == 'hard_route_violation',
        'physical_crash': race.get('terminal_reason') == 'physical_crash'
            or info.get('physical_crash_terminal') is not None,
        'research_outcome': outcome,
        'physical_crash_terminal': info.get('physical_crash_terminal'),
        'registered_crash_terminal_valid': bool(validate_registered_crash_terminal_info(info)),
    }

phase1 = json.loads((PHASE1 / 'smoke-summary.json').read_text(encoding='utf8'))
phase2b = json.loads((PHASE2B / 'eight-case-v4-profile-replay.json').read_text(encoding='utf8'))
phase1_by_id = {(r['model_id'], r['track_id']): r for r in phase1['runs']}
phase2b_by_id = {(r['model_id'], r['track_id']): r for r in phase2b['cases']}
asset_paths = {track: ROOT / 'assets' / 'tracks' / 'training' / f'{track}.json' for track in TRACK_IDS}
model_paths = {model: ROOT / 'models' / model / 'model-final.zip' for model, _ in SELECTED}
integrity_before = {
    'assets_sha256': {key: sha(path) for key, path in asset_paths.items()},
    'models_sha256': {key: sha(path) for key, path in model_paths.items()},
}
if integrity_before['assets_sha256'] != EXPECTED_ASSETS:
    raise RuntimeError(f'TRAIN asset SHA mismatch: {integrity_before["assets_sha256"]}')
if integrity_before['models_sha256'] != EXPECTED_MODELS:
    raise RuntimeError(f'frozen model SHA mismatch: {integrity_before["models_sha256"]}')
if phase1['source_commit'] != '6c479540134d98d7d27331879c926c8ba82c2965':
    raise RuntimeError('Phase 1 source baseline changed')
if (phase1['reset_seed'], phase1['step_limit'], phase1['inference_mode']) != (RESET_SEED, STEP_LIMIT, 'deterministic'):
    raise RuntimeError('Phase 1 reset/step/inference protocol mismatch')

registration = load_registration(ROOT, profile='spatial_rotating_owner_v3')
scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
if tuple(row.track_id for row in scenarios.scenarios) != TRACK_IDS:
    raise RuntimeError('Smoke scenario roster differs from the four TRAIN tracks')
scenario_by_id = {row.track_id: row for row in scenarios.scenarios}
models = {model_id: PPOModel.load(model_paths[model_id], device='cpu') for model_id, _ in SELECTED}
log_dir = EVIDENCE / 'episodes'
log_dir.mkdir(exist_ok=True)
rows = []
for model_id, training_seed_index in SELECTED:
    model = models[model_id]
    for track_id in TRACK_IDS:
        factory, profile_provenance = apply_runtime_profile(
            scenario_by_id[track_id].factory, FULL06_RUNTIME_PROFILE)
        effective = effective_runtime_profile(factory)
        if effective != profile_provenance['effective']:
            raise RuntimeError(f'profile provenance drift for {model_id}/{track_id}')
        env = factory.create()
        env.set_training_telemetry_enabled(True)
        case_id = (model_id, track_id)
        expected_p1 = phase1_by_id[case_id]
        log_path = log_dir / f'{model_id}__{track_id}.jsonl'
        timer_start = time.perf_counter()
        step_count = 0
        checkpoints = []
        collision_ids = set()
        final_info = {}
        classification = 'runtime_or_contract_exception'
        error = None
        route_highwater = 0.0
        action_count = 0
        with log_path.open('x', encoding='utf8', newline='\n') as stream:
            emit(stream, {
                'event': 'run_started', 'model_id': model_id,
                'training_seed_index': training_seed_index, 'track_id': track_id,
                'reset_seed': RESET_SEED, 'inference_mode': 'PPO deterministic=True',
                'step_limit': STEP_LIMIT, 'registration_profile': 'spatial_rotating_owner_v3',
                'requested_runtime_profile': profile_provenance['requested_profile'],
                'effective_runtime_profile': profile_provenance['effective'],
                'auto_drive': False, 'hybrid': False,
            })
            try:
                observation, reset_info = env.reset(seed=RESET_SEED)
                if not env.observation_space.contains(observation):
                    raise ValueError('reset observation violates observation_space')
                emit(stream, {
                    'event': 'reset', 'observation_shape': list(observation.shape),
                    'action_space_shape': list(env.action_space.shape),
                    'train_reset': reset_info.get('train_reset'),
                    'initial_position': reset_info.get('telemetry', {}).get('vehicle_position'),
                    'active_physics_version': reset_info.get('physics', {}).get('configuration', {}).get('ground_motion_version'),
                    'active_race_version': reset_info.get('race', {}).get('configuration', {}).get('version'),
                    'terminal_contract': reset_info.get('race', {}).get('physical_crash_contract'),
                    'wheel_support': wheel_support(reset_info),
                })
                for step in range(1, STEP_LIMIT + 1):
                    action = np.asarray(model.predict(observation, deterministic=True))
                    action_count += 1
                    try:
                        next_observation, reward, terminated, truncated, info = env.step(action)
                    except Exception as exc:
                        error = {
                            'class': type(exc).__name__, 'message': str(exc),
                            'failure_code': getattr(exc, 'failure_code', None),
                            'traceback': traceback.format_exc(),
                        }
                        last_outcome = safe(env.last_research_outcome)
                        classification = 'invalid_physical_transition' if (
                            isinstance(last_outcome, dict)
                            and last_outcome.get('kind') == 'invalid_physical_transition') else 'runtime_or_contract_exception'
                        emit(stream, {
                            'event': 'exception', 'attempt_step': step,
                            'action': action, 'exception': error,
                            'last_research_outcome': last_outcome,
                        })
                        break
                    step_count = step
                    final_info = info
                    race = info.get('race', {})
                    training = info.get('training_telemetry', {})
                    progress = float(info.get('route_highwater_meters', training.get('route_highwater_m', 0.0)) or 0.0)
                    route_highwater = max(route_highwater, progress)
                    checkpoint_id = info.get('completed_checkpoint_id')
                    if checkpoint_id is not None:
                        checkpoints.append(checkpoint_id)
                    hit_ids = info.get('telemetry', {}).get('obstacle_hit_ids', ())
                    collision_ids.update(hit_ids)
                    emit(stream, {
                        'event': 'step', 'step': step,
                        'action': action,
                        'controls': {'steering': float(action[0]), 'throttle': float(action[1]), 'brake': float(action[2])},
                        'reward': float(reward), 'route_highwater_m': progress,
                        'next_checkpoint_index': info.get('next_checkpoint_index'),
                        'completed_checkpoint_id': checkpoint_id,
                        'completed_checkpoint_ids': race.get('completed_checkpoint_ids', []),
                        'finish': {'finished': info.get('finished', False), 'finished_now': info.get('finished_now', False)},
                        'position': info.get('telemetry', {}).get('vehicle_position'),
                        'speed_mps': info.get('telemetry', {}).get('vehicle_speed'),
                        'wheel_support': wheel_support(info),
                        'collision': {
                            'obstacle_hit_ids': list(hit_ids),
                            'physical_crash_terminal': info.get('physical_crash_terminal'),
                        },
                        'terminal': {
                            'terminated': bool(terminated), 'truncated': bool(truncated),
                            **terminal_projection(info),
                        },
                    })
                    observation = next_observation
                    if info.get('finished'):
                        classification = 'finished'
                        break
                    if terminated:
                        reason = info.get('research_outcome', {}).get('reason')
                        classification = 'hard_route_violation' if reason == 'hard_route_violation' else (
                            'physical_crash' if reason == 'physical_crash' else 'valid_terminal_failure')
                        break
                    if truncated:
                        classification = 'step_limit' if info.get('research_outcome', {}).get('reason') == 'step_limit' else 'valid_truncation'
                        break
                else:
                    classification = 'step_limit'
            except Exception as exc:
                error = {'class': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
                last_outcome = safe(env.last_research_outcome)
                classification = 'invalid_physical_transition' if (
                    isinstance(last_outcome, dict)
                    and last_outcome.get('kind') == 'invalid_physical_transition') else 'runtime_or_contract_exception'
                emit(stream, {'event': 'exception', 'attempt_step': step_count + 1,
                              'exception': error, 'last_research_outcome': last_outcome})
            elapsed = time.perf_counter() - timer_start
            terminal = terminal_projection(final_info) if final_info else {}
            row = {
                'model_id': model_id, 'training_seed_index': training_seed_index,
                'track_id': track_id, 'reset_seed': RESET_SEED,
                'inference_mode': 'deterministic', 'step_limit': STEP_LIMIT,
                'requested_runtime_profile': profile_provenance['requested_profile'],
                'effective_runtime_profile': profile_provenance['effective'],
                'steps': step_count, 'action_count': action_count,
                'classification': classification,
                'finished': classification == 'finished',
                'checkpoints_completed': checkpoints,
                'next_checkpoint_index': final_info.get('next_checkpoint_index') if final_info else None,
                'route_highwater_m': route_highwater,
                'physical_crash': bool(terminal.get('physical_crash', False)),
                'physical_crash_terminal_valid': terminal.get('registered_crash_terminal_valid'),
                'hard_route_violation': bool(terminal.get('hard_route_violation', False)),
                'step_limit_reached': classification == 'step_limit',
                'invalid_physical_transition': classification == 'invalid_physical_transition',
                'terminal_reason': terminal.get('terminal_reason'),
                'research_outcome': terminal.get('research_outcome'),
                'physical_crash_terminal': terminal.get('physical_crash_terminal'),
                'collision_ids': sorted(collision_ids),
                'duration_seconds': round(elapsed, 3),
                'exception': error,
                'log': str(log_path),
                'phase1': {
                    'classification': expected_p1['classification'],
                    'steps': expected_p1['steps'],
                    'finished': expected_p1['finished'],
                    'checkpoints_completed': expected_p1['checkpoints_completed'],
                    'duration_seconds': expected_p1['duration_seconds'],
                    'log': expected_p1['log'],
                },
            }
            previous = phase2b_by_id.get(case_id)
            row['phase2b_v4_counterfactual'] = None if previous is None else {
                'outcome': previous['outcome'], 'step': previous['step'],
                'reason': previous.get('research_outcome', {}).get('reason'),
                'terminal_contract_valid': bool(previous.get('terminal_info', {}).get('physical_crash_terminal')),
                'contact_kind': previous.get('terminal_info', {}).get('physical_crash_terminal', {}).get('contact_kind'),
            }
            emit(stream, {'event': 'run_completed', **row})
        env.close()
        rows.append(row)
        print(json.dumps({
            'progress': True, 'completed_cases': len(rows), 'total_cases': 12,
            'model': model_id, 'track': track_id, 'classification': row['classification'],
            'steps': row['steps'], 'terminal_reason': row['terminal_reason'],
            'duration_seconds': row['duration_seconds'],
        }), flush=True)

integrity_after = {
    'assets_sha256': {key: sha(path) for key, path in asset_paths.items()},
    'models_sha256': {key: sha(path) for key, path in model_paths.items()},
}
if integrity_after != integrity_before:
    raise RuntimeError('model or TRAIN asset hashes changed during smoke')
counts = {}
for row in rows:
    counts[row['classification']] = counts.get(row['classification'], 0) + 1
summary = {
    'case_count': len(rows),
    'counts': counts,
    'finished': sum(row['finished'] for row in rows),
    'physical_crash': sum(row['physical_crash'] for row in rows),
    'valid_physical_crash': sum(row['physical_crash'] and row['physical_crash_terminal_valid'] for row in rows),
    'hard_route_violation': sum(row['hard_route_violation'] for row in rows),
    'step_limit': sum(row['step_limit_reached'] for row in rows),
    'invalid_physical_transition': sum(row['invalid_physical_transition'] for row in rows),
    'runtime_or_contract_exception': sum(row['classification'] == 'runtime_or_contract_exception' for row in rows),
    'all_effective_profiles_match': all(
        row['effective_runtime_profile'] == {
            'physics_version': 'spatial_road_edge_partial_lifecycle_v4',
            'race_version': 'wp7_directed_lifecycle_crash_contract_v4',
            'terminal_contract': 'wp3_physical_crash_terminal_v1',
            'max_abs_tilt_rad': 0.7,
        } for row in rows),
    'all_actions_logged': all(row['action_count'] == row['steps'] or row['classification'] == 'invalid_physical_transition' for row in rows),
    'model_asset_integrity_unchanged': integrity_before == integrity_after,
    'phase1_counts': {
        'finished': sum(r['classification'] == 'finished' for r in phase1['runs']),
        'invalid_physical_transition': sum(r['classification'] == 'invalid_physical_transition' for r in phase1['runs']),
    },
    'phase2b_invalid_case_replay_counts': phase2b['summary'],
    'rows': rows,
}
manifest = {
    'schema': 'f1rl_phase2c_deterministic_smoke_v1',
    'source_commit': '6c479540134d98d7d27331879c926c8ba82c2965',
    'target_worktree': str(ROOT),
    'profile': FULL06_RUNTIME_PROFILE,
    'runtime': {'python': platform.python_version(), 'device': 'cpu', 'torch_threads': torch.get_num_threads()},
    'protocol': {'registration_profile': 'spatial_rotating_owner_v3', 'reset_seed': RESET_SEED,
                 'step_limit': STEP_LIMIT, 'inference': 'deterministic',
                 'models': [model for model, _ in SELECTED], 'tracks': list(TRACK_IDS)},
    'phase2b_files_sha256': {
        rel: sha(ROOT / rel) for rel in [
            'f1rl/research/wp3_eval_only_coordinator.py',
            'scripts/wp3_eval_only_coordinator.py',
            'tests/research/test_wp3_eval_only_coordinator.py',
            'tests/research/test_wp3_smoke_runtime_profile.py',
        ]
    },
    'integrity_before': integrity_before,
    'integrity_after': integrity_after,
    'summary': summary,
}
manifest_path = EVIDENCE / 'phase2c-results.json'
manifest_path.write_text(json.dumps(safe(manifest), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
print(json.dumps({k: summary[k] for k in summary if k not in ('rows', 'phase2b_invalid_case_replay_counts')}, indent=2, ensure_ascii=False))
if len(rows) != 12 or not summary['all_effective_profiles_match'] or not summary['all_actions_logged'] or not summary['model_asset_integrity_unchanged']:
    raise SystemExit(4)

