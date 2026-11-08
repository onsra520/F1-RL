from __future__ import annotations
import dataclasses, hashlib, json, math, platform, sys, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r'D:\Projects\.worktrees\F1-RL\phase2b-smoke-profile-wiring').resolve()
BASELINE_MODELS = Path(r'D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit\models').resolve()
PHASE1 = Path(r'D:\Projects\.evidence\F1-RL\phase1-physics-v4-audit-20261106')
PHASE2A = Path(r'D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106')
OUT = Path(r'D:\Projects\.evidence\F1-RL\phase2b-smoke-profile-wiring-20261106')
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from f1rl.learning.algorithms.ppo import PPOModel
from f1rl.learning.crash_terminal_registered import validate_registered_crash_terminal_info
from f1rl.research.wp3_eval_only_coordinator import FULL06_RUNTIME_PROFILE, apply_runtime_profile
from f1rl.research.wp7_protocol import load_registration, prepare_scenarios

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

def sha(path):
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
    if isinstance(value, (np.floating, float)):
        n = float(value)
        return n if math.isfinite(n) else str(n)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, '__dict__'):
        return safe(vars(value))
    return str(value)

def vehicle_state(vehicle):
    q = vehicle.orientation
    return {
        'position_m': [float(vehicle.position.x), float(vehicle.position.y), float(vehicle.position.z)],
        'velocity_mps': [float(vehicle.velocity.x), float(vehicle.velocity.y), float(vehicle.velocity.z)],
        'speed_mps': float(vehicle.speed),
        'orientation_wxyz': [float(q.w), float(q.x), float(q.y), float(q.z)],
        'angular_velocity_radps': [float(vehicle.angular_velocity.x), float(vehicle.angular_velocity.y), float(vehicle.angular_velocity.z)],
        'grounded': bool(vehicle.grounded),
    }

phase1_summary = json.loads((PHASE1 / 'smoke-summary.json').read_text(encoding='utf8'))
phase1_invalid = [r for r in phase1_summary['runs'] if r['classification'] == 'invalid_physical_transition']
phase2a_v4 = json.loads((PHASE2A / 'v4-overlay-counterfactual.json').read_text(encoding='utf8'))
phase2a_by_id = {(r['model_id'], r['track_id']): r for r in phase2a_v4['cases']}
phase2a_v3 = json.loads((PHASE2A / 'v3-eight-case-instrumented-replay.json').read_text(encoding='utf8'))
phase2a_v3_by_id = {(r['model_id'], r['track_id']): r for r in phase2a_v3['cases']}
assert len(phase1_invalid) == len(phase2a_by_id) == 8

asset_paths = {track: ROOT / 'assets' / 'tracks' / 'training' / f'{track}.json' for track in EXPECTED_ASSETS}
model_paths = {model: BASELINE_MODELS / model / 'model-final.zip' for model in EXPECTED_MODELS}
integrity_before = {
    'assets_sha256': {key: sha(path) for key, path in asset_paths.items()},
    'models_sha256': {key: sha(path) for key, path in model_paths.items()},
}
assert integrity_before['assets_sha256'] == EXPECTED_ASSETS, integrity_before['assets_sha256']
assert integrity_before['models_sha256'] == EXPECTED_MODELS, integrity_before['models_sha256']

registration = load_registration(ROOT, profile='spatial_rotating_owner_v3')
scenarios = prepare_scenarios(ROOT, registration, seed_index=0)
scenario_by_track = {item.track_id: item for item in scenarios.scenarios}
models = {}
rows = []
for phase1_case in phase1_invalid:
    model_id, track_id = phase1_case['model_id'], phase1_case['track_id']
    model = models.get(model_id)
    if model is None:
        model = PPOModel.load(model_paths[model_id], device='cpu')
        models[model_id] = model
    factory, profile_provenance = apply_runtime_profile(
        scenario_by_track[track_id].factory, FULL06_RUNTIME_PROFILE)
    env = factory.create()
    env.set_training_telemetry_enabled(True)
    row = {
        'model_id': model_id,
        'track_id': track_id,
        'reset_seed': phase1_summary['reset_seed'],
        'inference_mode': 'deterministic',
        'requested_runtime_profile': profile_provenance['requested_profile'],
        'effective_runtime_profile': profile_provenance['effective'],
        'phase1_failed_attempt_step': phase1_case['steps'] + 1,
        'track_asset_sha256': integrity_before['assets_sha256'][track_id],
        'model_sha256': integrity_before['models_sha256'][model_id],
        'outcome': 'not_started',
    }
    try:
        obs, reset_info = env.reset(seed=phase1_summary['reset_seed'])
        row['reset_runtime_telemetry'] = safe({
            'physics': reset_info.get('physics', {}).get('configuration', {}),
            'race': reset_info.get('race', {}).get('configuration', {}),
            'physical_crash_contract': reset_info.get('race', {}).get('physical_crash_contract'),
        })
        for step in range(1, phase1_summary['step_limit'] + 1):
            before = env._simulation.snapshot
            action = model.predict(obs, deterministic=True)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                row.update({
                    'outcome': 'exception', 'step': step,
                    'pre_transition_vehicle': vehicle_state(before.world.vehicle),
                    'action': safe(action),
                    'exception': {'class': type(exc).__name__, 'message': str(exc),
                                  'failure_code': getattr(exc, 'failure_code', None),
                                  'traceback': traceback.format_exc()},
                    'research_outcome': safe(env.last_research_outcome),
                })
                break
            if terminated or truncated:
                row.update({
                    'outcome': 'terminal', 'step': step,
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'terminal_reason': info.get('race', {}).get('terminal_reason'),
                    'terminal_info': safe(info),
                    'physical_crash_contract_valid': bool(validate_registered_crash_terminal_info(info)),
                    'final_vehicle': vehicle_state(env._simulation.snapshot.world.vehicle),
                    'research_outcome': safe(env.last_research_outcome),
                })
                break
        else:
            row['outcome'] = 'step_limit'
    except Exception as exc:
        row.update({'outcome': 'driver_error', 'exception': {
            'class': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}})
    finally:
        env.close()
    previous = phase2a_by_id[(model_id, track_id)]
    row['phase2a_v4_counterfactual'] = {
        'outcome': previous['outcome'], 'step': previous['step'],
        'terminated': previous.get('terminated'), 'truncated': previous.get('truncated'),
        'reason': previous.get('research_outcome', {}).get('reason'),
        'physical_crash_contract_valid': bool(previous.get('terminal_info', {}).get('physical_crash_terminal')),
    }
    row['episode_equivalence'] = {
        'same_model_track_reset_seed_and_deterministic_mode': True,
        'same_terminal_outcome_and_step_as_phase2a_v4': (
            row.get('outcome') == previous['outcome'] and row.get('step') == previous['step']
            and row.get('research_outcome', {}).get('reason') == previous.get('research_outcome', {}).get('reason')),
        'transition_substep_equivalence_claimed': False,
        'note': 'v4 dynamics differ from v3 from the first transition; v3 failed-attempt substep is not a same-state replay target.',
    }
    rows.append(row)

integrity_after = {
    'assets_sha256': {key: sha(path) for key, path in asset_paths.items()},
    'models_sha256': {key: sha(path) for key, path in model_paths.items()},
}
assert integrity_after == integrity_before

summary = {
    'case_count': len(rows),
    'terminal_physical_crash_valid_count': sum(
        r.get('outcome') == 'terminal' and r.get('terminated') is True
        and r.get('truncated') is False and r.get('terminal_reason') == 'physical_crash'
        and r.get('physical_crash_contract_valid') is True
        and r.get('research_outcome', {}).get('valid_sample') is True
        and r.get('research_outcome', {}).get('reason') == 'physical_crash'
        for r in rows),
    'invalid_or_exception_count': sum(r.get('outcome') != 'terminal' for r in rows),
    'episode_equivalence_count': sum(r['episode_equivalence']['same_terminal_outcome_and_step_as_phase2a_v4'] for r in rows),
    'transition_substep_equivalence_claimed': False,
    'all_requested_profiles_equal_effective': all(
        r['requested_runtime_profile'] == FULL06_RUNTIME_PROFILE
        and r['effective_runtime_profile'] == {
            'physics_version': 'spatial_road_edge_partial_lifecycle_v4',
            'race_version': 'wp7_directed_lifecycle_crash_contract_v4',
            'terminal_contract': 'wp3_physical_crash_terminal_v1',
            'max_abs_tilt_rad': 0.7,
        } for r in rows),
    'asset_model_integrity_unchanged': integrity_before == integrity_after,
    'rows': [{
        'model_id': r['model_id'], 'track_id': r['track_id'],
        'phase1_failed_attempt_step': r['phase1_failed_attempt_step'],
        'phase2b_terminal_step': r.get('step'), 'outcome': r.get('outcome'),
        'terminal_reason': r.get('terminal_reason'),
        'valid_physical_crash_contract': r.get('physical_crash_contract_valid'),
        'episode_equivalent_to_phase2a_v4': r['episode_equivalence']['same_terminal_outcome_and_step_as_phase2a_v4'],
        'exception': r.get('exception', {}).get('message'),
    } for r in rows],
}
result = {
    'schema': 'f1rl_phase2b_eight_case_profile_replay_v1',
    'source_baseline': '6c479540134d98d7d27331879c926c8ba82c2965',
    'phase2b_source_head': __import__('subprocess').check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    'runtime': {'python': platform.python_version(), 'torch_threads': torch.get_num_threads(),
                'device': 'cpu'},
    'profile': FULL06_RUNTIME_PROFILE,
    'phase1_replay_identity': {
        'source_commit': phase1_summary['source_commit'], 'reset_seed': phase1_summary['reset_seed'],
        'step_limit': phase1_summary['step_limit'], 'inference_mode': phase1_summary['inference_mode'],
        'invalid_cases': [{'model_id': c['model_id'], 'track_id': c['track_id'],
                           'failed_attempt_step': c['steps'] + 1} for c in phase1_invalid],
    },
    'integrity_before': integrity_before,
    'integrity_after': integrity_after,
    'summary': summary,
    'cases': rows,
}
path = OUT / 'eight-case-v4-profile-replay.json'
path.write_text(json.dumps(safe(result), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
if summary['terminal_physical_crash_valid_count'] != 8 or summary['invalid_or_exception_count'] != 0:
    raise SystemExit(2)
if summary['episode_equivalence_count'] != 8 or not summary['asset_model_integrity_unchanged']:
    raise SystemExit(3)


