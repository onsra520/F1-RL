"""Prepare, execute (parallel by default) one paired internal TRAIN trial. No signature on the kiet branch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from f1rl.research.wp7_protocol import _unique
from scripts.wp7_penalty_trial_protocol import build_proposal, validate_proposal
from scripts.wp7_penalty_trial_runner import INTERVENTIONS, _require_clean_source, prepare_output, run_dir_name, write_training_result


def external(root, raw):
    path = Path(raw).resolve()
    if path == root or path.is_relative_to(root):
        raise ValueError('path must be outside checkout')
    return path


def _load_plan(root, raw):
    plan = json.loads(external(root, raw).read_text(encoding='utf8'), object_pairs_hook=_unique)
    if set(plan) != {'proposal', 'output_root'}:
        raise ValueError('invalid plan')
    return plan['proposal'], external(root, plan['output_root'])


def _child_env():
    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    return env


def _run_pool(root, jobs, tasks, logs):
    """Run child commands with at most `jobs` at once. A failing child does not stop the others."""
    pending, running, failed = list(tasks), [], []
    while pending or running:
        while pending and len(running) < jobs:
            name, cmd = pending.pop(0)
            log = (logs / f'{name}.log').open('x', encoding='utf8')
            running.append((name, subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=_child_env()), log))
        still = []
        for name, proc, log in running:
            code = proc.poll()
            if code is None:
                still.append((name, proc, log))
            else:
                log.close()
                if code != 0:
                    failed.append((name, code))
        running = still
        if running:
            time.sleep(1)
    return failed


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', required=True); prep.add_argument('--plan', required=True); prep.add_argument('--evidence', required=True)
    execute = sub.add_parser('execute'); execute.add_argument('--plan', required=True); execute.add_argument('--evidence', required=True); execute.add_argument('--jobs', type=int, default=12)
    worker = sub.add_parser('run-one'); worker.add_argument('--plan', required=True); worker.add_argument('--evidence', required=True)
    worker.add_argument('--intervention', required=True); worker.add_argument('--cohort', required=True); worker.add_argument('--replicate', type=int, required=True)
    shard = sub.add_parser('evaluate-shard'); shard.add_argument('--plan', required=True); shard.add_argument('--evidence', required=True)
    shard.add_argument('--cohort', required=True); shard.add_argument('--replicate', type=int, required=True); shard.add_argument('--track', required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(strict=True)
    if args.action == 'prepare':
        _require_clean_source(root)
        output, plan = external(root, args.output), external(root, args.plan)
        if output.exists() or plan.exists(): raise ValueError('new output and plan paths required')
        from scripts.wp7_train5_batch import preflight_checkpoints
        preflight_checkpoints(root, Path(args.evidence))
        proposal = build_proposal(root)
        with plan.open('x', encoding='utf8') as f:
            json.dump({'proposal': proposal, 'output_root': str(output)}, f, indent=2); f.write('\n')
        print('Plan saved (no signature):', plan)
        return
    proposal, output = _load_plan(root, args.plan)
    evidence = Path(args.evidence)
    if args.action == 'run-one':
        from scripts.wp7_penalty_trial_runner import run_one
        run_one(root=root, proposal=proposal, output=output, evidence_root=evidence,
                intervention=args.intervention, cohort=args.cohort, replicate=args.replicate)
    elif args.action == 'evaluate-shard':
        from scripts.wp7_penalty_trial_evaluate import evaluate_shard
        n = evaluate_shard(root=root, output=output, proposal=proposal, evidence_root=evidence,
                           cohort=args.cohort, replicate=args.replicate, track=args.track)
        print(f'Shard {args.cohort} {args.replicate} {args.track}: {n} records', flush=True)
    else:
        from scripts.wp7_penalty_trial_evaluate import evaluate_shard, merge_shards, shard_rows
        validate_proposal(root, proposal)
        out = prepare_output(root=root, proposal=proposal, output=output)
        logs = out / 'logs'
        logs.mkdir()
        plan_path = external(root, args.plan)
        base = [sys.executable, '-B', '-m', 'scripts.wp7_penalty_trial_cli']
        tasks = [(run_dir_name(iv, row['cohort'], row['replicate']),
                  base + ['run-one', '--plan', str(plan_path), '--evidence', str(evidence), '--intervention', iv,
                          '--cohort', row['cohort'], '--replicate', str(row['replicate'])])
                 for iv in INTERVENTIONS for row in proposal['pairs']]
        failed = _run_pool(root, args.jobs, tasks, logs)
        if failed:
            raise SystemExit(f'training workers failed: {failed}')
        write_training_result(root=root, proposal=proposal, output=out)
        shards = sorted({(r[0], r[1], r[2]) for r in proposal['evaluation_seed_rows']})
        tasks = [(f'eval-{c}-{rep}-{t}', base + ['evaluate-shard', '--plan', str(plan_path), '--evidence', str(evidence),
                                                  '--cohort', c, '--replicate', str(rep), '--track', t])
                 for c, rep, t in shards]
        failed = _run_pool(root, args.jobs, tasks, logs)
        if failed:
            raise SystemExit(f'evaluation workers failed: {failed}')
        merge_shards(root=root, output=out, proposal=proposal)


if __name__ == '__main__':
    main()
