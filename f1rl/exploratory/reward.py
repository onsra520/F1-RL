"""Versioned demo-only curriculum; never changes registered Gym/physics outcomes."""
from math import isfinite

from ..sandbox.independent_suspension_dynamics import SuspensionResolutionError

REGISTERED = {'version': 'registered'}


def validate(value):
    if value == REGISTERED:
        return dict(REGISTERED)
    expected = {'version', 'training_corridor_m', 'lateral_penalty_per_m2_s', 'terminal_penalty'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('unsupported demo-only reward/curriculum version')
    corridor_by_version = {
        'straight_route_stability_demo_v1': 2.,
        'straight_route_stability_demo_v2': 3.,
    }
    if (value['version'] not in corridor_by_version
            or type(value['training_corridor_m']) not in (int, float)
            or value['training_corridor_m'] != corridor_by_version[value['version']]):
        raise ValueError('unsupported demo-only reward/curriculum version')
    for name, maximum in (('lateral_penalty_per_m2_s', 10.), ('terminal_penalty', 100.)):
        number = value[name]
        if type(number) not in (int, float) or not isfinite(number) or not 0. < number <= maximum:
            raise ValueError('demo reward parameter outside its bounded scope')
    return dict(value)


def apply_training_curriculum(env, result, config):
    if config['version'] == 'registered':
        return result
    observation, registered_reward, terminated, truncated, original_info = result
    distance = env._authored_route.horizontal_distance(env._simulation.snapshot.world.vehicle.position)
    stop = distance > config['training_corridor_m'] and not (terminated or truncated)
    original_failure = original_info.get('research_outcome', {}).get('reason') == 'hard_route_violation'
    lateral_penalty = config['lateral_penalty_per_m2_s']*distance*distance*env._simulation_config.fixed_dt
    terminal_penalty = config['terminal_penalty'] if stop or original_failure else 0.
    reward = float(registered_reward)-lateral_penalty-terminal_penalty
    info = {**original_info, 'demo_training': {
        'version': config['version'], 'registered_reward': float(registered_reward),
        'training_corridor_m': float(config['training_corridor_m']),
        'lateral_distance_m': distance, 'lateral_penalty': lateral_penalty,
        'terminal_penalty': terminal_penalty, 'reward': reward,
        'curriculum_terminal': stop, 'scientific_evidence': False}}
    return observation, reward, terminated or stop, truncated, info


def apply_training_physics_failure(env, error, config):
    """End a tuning episode at its last certified state on unsupported contact."""
    config = validate(config)
    if (not isinstance(error, SuspensionResolutionError)
            or error.failure_code != 'unsupported_contact'
            or config['version'] == 'registered'):
        raise ValueError('only the versioned tuning profile may terminate unsupported contact')
    snapshot = env._simulation.snapshot
    progress = env._progress
    if progress is None or not env._needs_reset:
        raise RuntimeError('physics failure did not leave a terminal last-certified environment state')
    vehicle = snapshot.world.vehicle
    observation = env._encode_observation(vehicle, env._route_target(vehicle, progress))
    distance = env._authored_route.horizontal_distance(vehicle.position)
    lateral_penalty = config['lateral_penalty_per_m2_s'] * distance * distance * env._simulation_config.fixed_dt
    terminal_penalty = config['terminal_penalty']
    reward = -lateral_penalty - terminal_penalty
    info = {
        'tick': snapshot.tick, 'elapsed_time': snapshot.elapsed_time,
        'next_checkpoint_index': progress.next_checkpoint_index,
        'completed_checkpoint_id': None, 'finished': progress.finished,
        'finished_now': False, 'game_score': snapshot.world.score.points,
        'reward_components': {}, **env._physics_info(snapshot),
        **env._train_reset_info(), **env._race_info(),
        'route_highwater_meters': env._route_highwater,
        **env._training_state_info(snapshot, progress, env._route_highwater),
        'research_outcome': env.last_research_outcome,
        'demo_training': {
            'version': config['version'], 'registered_reward': 0.0,
            'training_corridor_m': float(config['training_corridor_m']),
            'lateral_distance_m': distance, 'lateral_penalty': lateral_penalty,
            'terminal_penalty': terminal_penalty, 'reward': reward,
            'curriculum_terminal': False,
            'physics_abort_terminal': True,
            'physics_failure_code': error.failure_code,
            'physics_failure_detail': str(error),
            'scientific_evidence': False,
        },
    }
    return observation, reward, True, False, info
