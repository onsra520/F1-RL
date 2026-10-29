"""TRAIN-only diagnostics. The diagnostic controller is scripted, never PPO."""
import argparse
import cProfile
from dataclasses import asdict, replace
from hashlib import sha256
import json
from math import atan2
from pathlib import Path
import pstats
import time

ROOT = Path(__file__).resolve().parents[2]


def diagnostic_action(observation, yaw_rate):
    import numpy as np
    heading_error = atan2(float(observation[10]), float(observation[12]))
    return np.array([np.clip(1.5*heading_error - .4*yaw_rate, -1., 1.), .6, 0.], dtype=np.float32)


def trace_episode(factory, model, seed, controller, output):
    from src.simulation.geometry.vector import Vec3
    env = replace(factory, reset_config=replace(factory.reset_config, seed=seed)).create()
    started = time.perf_counter()
    observation, info = env.reset(seed=seed)
    reset_seconds = time.perf_counter()-started
    total, steps, previous_speed = 0., 0, env._simulation.snapshot.world.vehicle.speed
    trace = sha256()
    with output.open('x', encoding='utf-8') as stream:
        while True:
            before = env._simulation.snapshot.world.vehicle
            action = diagnostic_action(observation, before.angular_velocity.y) if controller else model.predict(observation)
            action_observation = observation.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += reward
            vehicle = env._simulation.snapshot.world.vehicle
            forward = vehicle.orientation.rotate(Vec3(0., 0., 1.))
            row = {'step': steps, 'action': action.tolist(), 'observation': action_observation.tolist(),
                'vehicle': asdict(vehicle), 'heading_rad': atan2(forward.x, forward.z),
                'longitudinal_speed_mps': vehicle.velocity.dot(forward),
                'speed_acceleration_mps2': (vehicle.speed-previous_speed)/env._simulation_config.fixed_dt,
                'route_distance_m': env._authored_route.horizontal_distance(vehicle.position),
                'reward': reward, 'info': info}
            content = json.dumps(row, sort_keys=True, allow_nan=False)
            stream.write(content+'\n')
            trace.update(content.encode())
            previous_speed = vehicle.speed
            if terminated or truncated:
                break
    env.close()
    return {'policy': 'scripted_diagnostic_not_PPO' if controller else 'original_PPO',
        'seed': seed, 'steps': steps, 'return': total, 'reset_seconds': reset_seconds,
        'wall_seconds': time.perf_counter()-started, 'progress': info.get('route_highwater_meters',0.)/env._authored_route.total_meters,
        'outcome': info.get('research_outcome'), 'final_action': action.tolist(),
        'final_position': asdict(vehicle.position), 'final_heading_rad': row['heading_rad'],
        'final_yaw_rate': vehicle.angular_velocity.y, 'final_route_distance_m': row['route_distance_m'],
        'trace_sha256': trace.hexdigest()}


def benchmark(factory, model, threads, output, *, torch_runtime):
    import numpy as np
    rows = []
    for count in threads:
        torch_runtime.set_num_threads(count)
        env = replace(factory, reset_config=replace(factory.reset_config, seed=10000)).create()
        started = time.perf_counter()
        observation, _ = env.reset(seed=10000)
        reset_seconds = time.perf_counter()-started
        profiler = cProfile.Profile()
        trace = sha256()
        profiler.enable()
        started = time.perf_counter()
        for index in range(80):
            result = env.step(np.array([0., .6, 0.], dtype=np.float32))
            trace.update(json.dumps([result[0].tolist(), result[1], result[2], result[3]], allow_nan=False).encode())
            if result[2] or result[3]:
                raise RuntimeError('benchmark fixed workload ended unexpectedly')
        step_seconds = time.perf_counter()-started
        profiler.disable()
        stats = pstats.Stats(profiler)
        top = [{'file': Path(file).as_posix(), 'line': line, 'function': name,
                'calls': value[1], 'self_seconds': value[2], 'cumulative_seconds': value[3]}
            for (file,line,name), value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:30]]
        sensor_seconds = sum(value[3] for (file,_,name),value in stats.stats.items() if file.endswith('sensor_suite.py') and name=='sense')
        physics_seconds = sum(value[3] for (file,_,name),value in stats.stats.items() if file.endswith('simulation.py') and name=='step')
        started = time.perf_counter()
        for _ in range(80): json.dumps(result[4], sort_keys=True, allow_nan=False)
        logging_seconds = time.perf_counter()-started
        started = time.perf_counter()
        model.save(output/f'benchmark-serialization-{count}.zip')
        serialization_seconds = time.perf_counter()-started
        env.close()
        rows.append({'threads':count,'steps':80,'reset_seconds':reset_seconds,
            'env_step_seconds_profiled':step_seconds,'sensor_seconds':sensor_seconds,
            'physics_contact_seconds':physics_seconds,'json_logging_seconds_80':logging_seconds,
            'checkpoint_seconds':serialization_seconds,'trace_sha256':trace.hexdigest(),'top_functions':top})
    if len({row['trace_sha256'] for row in rows}) != 1:
        raise RuntimeError('thread benchmark changed fixed-input physics/observations/rewards')
    return rows


def main(*, torch_factory):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+')
    args = parser.parse_args()
    torch = torch_factory()
    from src.exploratory.config import resolve
    from src.rl.algorithms.ppo import PPOModel
    from src.validation.wp7_protocol import load_registration, prepare_scenarios
    settings = resolve(ROOT)
    torch.set_num_threads(settings['limits']['threads'])
    torch.set_num_interop_threads(settings['limits']['interop_threads'])
    args.output.mkdir(parents=True, exist_ok=False)
    model = PPOModel.load(args.model)
    factory = prepare_scenarios(ROOT, load_registration(ROOT,profile=settings['profile']),seed_index=0).scenarios[0].factory
    assert factory.reset_track_id == 'straight-sprint-v1'
    rows = []
    for controller in (False, True):
        for seed in (10000,10001):
            name = 'scripted' if controller else 'original-ppo'
            row = trace_episode(factory,model,seed,controller,args.output/f'{name}-{seed}.jsonl')
            rows.append(row)
            print(json.dumps(row),flush=True)
    profiling = benchmark(factory, model, args.threads or [settings['limits']['threads']], args.output, torch_runtime=torch)
    summary = {'track':'straight-sprint-v1','diagnostic_seeds':[10000,10001],
        'reserved_evaluation_seeds':list(range(11000,11006)),
        'episodes':rows,'profiling':profiling,'PPO_training_executed':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'profile':[ {key:value for key,value in row.items() if key!='top_functions'} for row in profiling]}))


if __name__ == '__main__':
    main()
