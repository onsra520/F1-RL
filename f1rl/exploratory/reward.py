"""Versioned demo-only curriculum; never changes registered Gym/physics outcomes."""
from math import isfinite

REGISTERED = {'version': 'registered'}


def validate(value):
    if value == REGISTERED:
        return dict(REGISTERED)
    expected = {'version', 'training_corridor_m', 'lateral_penalty_per_m2_s', 'terminal_penalty'}
    if (not isinstance(value, dict) or set(value) != expected
            or value['version'] != 'straight_route_stability_demo_v1'
            or type(value['training_corridor_m']) not in (int, float) or value['training_corridor_m'] != 2.):
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
        'lateral_distance_m': distance, 'lateral_penalty': lateral_penalty,
        'terminal_penalty': terminal_penalty, 'reward': reward,
        'curriculum_terminal': stop, 'scientific_evidence': False}}
    return observation, reward, terminated or stop, truncated, info
