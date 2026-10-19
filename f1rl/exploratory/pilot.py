"""Bounded Straight-only pilot. Called only after independent execution consent."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .permit import Budget, PilotStopped, Session, authorize, read_json, _utc


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def reserve(permit):
    output = Path(permit['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'lease.json', dict(permit_sha256=permit['sha256'], started_epoch=time.time()))
    return output


def claim(permit):
    """Return a fixed monotonic deadline; initialization cannot extend it."""
    output = Path(permit['output_directory'])
    lease = read_json(output/'lease.json')
    if (set(lease) != {'permit_sha256', 'started_epoch'} or lease['permit_sha256'] != permit['sha256']
            or type(lease['started_epoch']) not in (int, float)
            or not 0 <= time.time()-lease['started_epoch'] < permit['limits']['wall_seconds']):
        raise ValueError('invalid, future or expired execution lease')
    deadline = time.monotonic()+lease['started_epoch']+permit['limits']['wall_seconds']-time.time()-.05
    with (output/'worker.claim').open('x', encoding='utf-8') as stream:
        json.dump(dict(permit_sha256=permit['sha256'], pid=os.getpid()), stream)
    return deadline


def supervise(root, permit, envelope):
    """Parent enforces a hard process deadline including all baselines/evaluations."""
    started = time.monotonic()
    authorize(root, permit, envelope)
    wall_seconds = permit['limits']['wall_seconds']
    if time.monotonic()-started >= wall_seconds:
        raise PilotStopped('pilot deadline reached during authorization')
    output = reserve(permit)
    write_json(output/'permit.json', permit)
    write_json(output/'authorization.json', envelope)
    write_json(output/'supervisor.json', dict(status='running', training_authorized=False))
    thread_limit = str(permit['limits']['threads'])
    blas_limit = str(permit['limits']['blas_threads'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=thread_limit, MKL_NUM_THREADS=blas_limit,
        OPENBLAS_NUM_THREADS=blas_limit, NUMEXPR_NUM_THREADS=blas_limit, VECLIB_MAXIMUM_THREADS=blas_limit,
        PYTHONPATH=str(Path(root).resolve()), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(Path(root)/'scripts/exploratory_ppo.py'),
        '_worker', '--permit', str(output/'permit.json'), '--authorization', str(output/'authorization.json')]
    try:
        with (output/'worker.stdout.log').open('w', encoding='utf-8') as stdout, (output/'worker.stderr.log').open('w', encoding='utf-8') as stderr:
            remaining = wall_seconds-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, wall_seconds)
            result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr,
                timeout=remaining, check=False)
        code = result.returncode
        status = 'completed' if code == 0 else 'worker_failed'
    except subprocess.TimeoutExpired:
        code, status = 124, 'timeout'
    except BaseException:
        write_json(output/'supervisor.json', dict(status='interrupted', training_authorized=False))
        raise
    write_json(output/'supervisor.json', dict(status=status, returncode=code,
        elapsed_seconds=time.monotonic()-started, training_authorized=False,
        scientific_acceptance=False))
    return code


def evaluate_episode(session, model, name, episode):
    import numpy as np
    env = session.create_env(learning=False, episode=episode)
    seed = session.permit['seeds']['evaluation'][episode]
    rng = np.random.default_rng(seed)
    trace = sha256()
    total, collisions, off_route, steps = 0., 0, 0, 0
    completed_checkpoint_ids, directed_crossings = [], []
    info, error = {}, None
    try:
        observation, _ = env.reset(seed=seed)
        while True:
            action = (rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
                      if name == 'random' else model.predict(observation, deterministic=True))
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += float(reward)
            race = info.get('race') or {}
            completed_checkpoint_ids.extend(race.get('completed_checkpoint_ids', ()))
            directed_crossings.extend(race.get('directed_crossings', ()))
            collisions += len(info['telemetry']['obstacle_hit_ids'])
            off_route += int(info.get('race', {}).get('terminal_failure') is True)
            trace.update(json.dumps(dict(action=action.tolist(), observation=observation.tolist(),
                reward=float(reward), terminated=terminated, truncated=truncated, info=info),
                sort_keys=True, allow_nan=False).encode())
            if terminated or truncated:
                break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        env.close()
    outcome = (getattr(env, 'last_research_outcome', None) if error else info.get('research_outcome')) or {}
    valid = error is None and outcome.get('valid_sample') is True
    return dict(policy=name, episode=episode, environment_seed=seed, valid_sample=valid,
        success=valid and outcome.get('kind') == 'successful_finish',
        progress=min(1., info.get('route_highwater_meters', 0.)/env._authored_route.total_meters),
        completed_checkpoint_ids=completed_checkpoint_ids, directed_crossings=directed_crossings,
        collisions=collisions, off_route_events=off_route, duration_s=info.get('elapsed_time', 0.),
        steps=steps, **{'return':total}, outcome=outcome, error=error, trace_sha256=trace.hexdigest())


def evaluate_roster(session, model, name, output):
    rows = []
    for episode in range(session.permit['evaluation']['episodes_per_policy']):
        row = evaluate_episode(session, model, name, episode)
        append_json(Path(output)/'evaluation.jsonl', row)
        if row.get('valid_sample') is False:
            raise RuntimeError(f'invalid/aborted evaluation episode {episode}: {row.get("error") or row.get("outcome")}')
        rows.append(row)
    return rows


def save_checkpoint(model, output, name, updates, optimizer_steps):
    temporary = output/(name+'.pending.zip')
    final = output/(name+'.zip')
    model.save(temporary)
    os.replace(temporary, final)
    append_json(output/'artifacts.jsonl', dict(path=final.name,
        sha256=sha256(final.read_bytes()).hexdigest(), timesteps=model.num_timesteps,
        ppo_epoch_updates=updates, optimizer_steps=optimizer_steps))
    return final


def weights_sha256(model):
    value = sha256()
    for name, tensor in sorted(model._model.policy.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def worker(root, permit, envelope):
    """No auto-resume. A consumed run requires a fresh owner-bound permit."""
    authorize(root, permit, envelope)
    deadline = claim(permit)
    expiry_remaining = (_utc(envelope['payload']['expires_at_utc'])-datetime.now(timezone.utc)).total_seconds()
    budget = Budget(deadline=min(deadline, time.monotonic()+expiry_remaining-.05),
                    learning_limit=permit['limits']['learning_timesteps'])
    import torch
    torch.set_num_threads(permit['limits']['threads'])
    torch.set_num_interop_threads(permit['limits']['interop_threads'])
    torch.use_deterministic_algorithms(True)
    from stable_baselines3.common.logger import configure
    from ..learning.algorithms.ppo import PPOConfig, PPOModel

    output = Path(permit['output_directory'])
    model = env = hook = None
    optimizer_steps = 0
    phase = 'initialization'

    def state(status, **extra):
        write_json(output/'state.json', dict(status=status, phase=phase,
            learning_timesteps=0 if model is None else model.num_timesteps,
            environment_learning_steps=budget.learning_steps, optimizer_steps=optimizer_steps,
            training_authorized=False, scientific_evidence=False, **extra))

    def episode_row(step, reward, info):
        append_json(output/'learning-episodes.jsonl', dict(step=step, episode_return=reward,
            outcome=info.get('research_outcome'), progress_m=info.get('route_highwater_meters')))

    try:
        session = Session(root, permit, envelope, budget, episode_observer=episode_row)
        env = session.create_env(learning=True)
        config = PPOConfig(**permit['ppo'])
        model = PPOModel.create(env, config)
        model._model.set_logger(configure(str(output/'learning'), ['csv']))
        before = weights_sha256(model)
        save_checkpoint(model, output, 'untrained', 0, 0)
        phase = 'random_baseline'
        state('running')
        random_rows = evaluate_roster(session, model, 'random', output)
        phase = 'untrained_baseline'
        state('running')
        untrained_rows = evaluate_roster(session, model, 'untrained', output)
        phase = 'learning'
        state('running')

        def optimizer_step(*args, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            budget.check()

        hook = model._model.policy.optimizer.register_step_post_hook(optimizer_step)

        def progress(step):
            budget.check()
            if step % permit['logging']['progress_every_steps'] == 0:
                append_json(output/'learning-steps.jsonl', dict(step=step,
                    ppo_epoch_updates=model._model._n_updates, optimizer_steps=optimizer_steps))
                state('running')
            if step % permit['limits']['checkpoint_steps'] == 0:
                session.validate()
                save_checkpoint(model, output, f'checkpoint-{step}', model._model._n_updates, optimizer_steps)

        model.learn(permit['limits']['learning_timesteps'], progress_hook=progress)
        hook.remove()
        hook = None
        after = weights_sha256(model)
        if model.num_timesteps != permit['limits']['learning_timesteps'] or optimizer_steps <= 0 or before == after:
            raise RuntimeError('genuine learning evidence missing')
        final = save_checkpoint(model, output, 'trained-final', model._model._n_updates, optimizer_steps)
        phase = 'trained_evaluation'
        state('running')
        loaded = PPOModel.load(final, device=permit['limits']['device'])
        trained_rows = evaluate_roster(session, loaded, 'trained_final', output)
        phase = 'replay'
        replay_index = permit['evaluation']['replay_episode']
        replay = evaluate_episode(session, loaded, 'trained_final', replay_index)
        write_json(output/'replay.json', dict(equal=replay == trained_rows[replay_index], episode=replay))
        if replay != trained_rows[replay_index]:
            raise RuntimeError('deterministic replay mismatch')
        write_json(output/'comparison.json', dict(random=random_rows, untrained=untrained_rows,
            trained_final=trained_rows, initial_weights_sha256=before, final_weights_sha256=after,
            optimizer_steps=optimizer_steps, ppo_epoch_updates=model._model._n_updates,
            timesteps=model.num_timesteps, training_authorized=False, scientific_evidence=False,
            learning_proven=True, autonomous_completion_observed=any(r['success'] for r in trained_rows)))
        state('completed', initial_weights_sha256=before, final_weights_sha256=after)
        return 0
    except BaseException as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}',
              last_learning_outcome=None if env is None else getattr(env, 'last_research_outcome', None))
        raise
    finally:
        if hook is not None:
            hook.remove()
        if env is not None:
            env.close()
