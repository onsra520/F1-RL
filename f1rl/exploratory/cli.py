"""Prepare or execute a bounded local run with explicit engineering-only intent."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preflight = commands.add_parser('preflight', help='inspect source/config; no Gym reset or PPO model')
    preflight.add_argument('--output-directory', type=Path, required=True)
    preflight.add_argument('--permit', type=Path, required=True)
    execute = commands.add_parser('execute', help='run locally with explicit engineering-only intent')
    execute.add_argument('--permit', type=Path, required=True)
    execute.add_argument('--engineering-only', action='store_true')
    execute.add_argument('--source-commit-sha', required=True)
    execute.add_argument('--config', type=Path, required=True)
    execute.add_argument('--seed', type=int, required=True)
    execute.add_argument('--output-directory', type=Path, required=True)
    worker = commands.add_parser('_worker')
    worker.add_argument('--permit', type=Path, required=True)
    worker.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    from .permit import prepare, read_json
    trace_id = uuid4().hex
    try:
        if args.command == 'preflight':
            permit = prepare(root, args.output_directory)
            if args.output_directory.exists():
                raise ValueError('execution output must not already exist')
            if args.permit.resolve().is_relative_to(root):
                raise ValueError('prepared permit must be external to source checkout')
            args.permit.parent.mkdir(parents=True, exist_ok=True)
            with args.permit.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(permit, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(trace_id=trace_id, permit_sha256=permit['sha256'], state=permit['state'],
                exploratory_execution_authorized=False, training_authorized=False)))
            return 0
        from .pilot import supervise, worker
        permit = read_json(args.permit)
        if args.command == 'execute':
            if not args.engineering_only:
                raise ValueError('explicit --engineering-only intent is required')
            config = args.config.resolve(strict=True)
            if config != (root/'config.yml').resolve(strict=True):
                raise ValueError('engineering config must be the prepared repository config.yml')
            envelope = dict(execution_mode='engineering_only',
                source_commit_sha=args.source_commit_sha,
                config_sha256=sha256(config.read_bytes()).hexdigest(), seed=args.seed,
                output_directory=str(args.output_directory.resolve()), trace_id=trace_id)
            result = supervise(root, permit, envelope)
            print(json.dumps(dict(trace_id=trace_id, module='exploratory',
                operation='local_ppo_execution', outcome='completed' if result == 0 else 'failed',
                engineering_only=True, scientific_evidence=False)))
            return result
        envelope = read_json(args.authorization)
        return worker(root, permit, envelope)
    except (ValueError, OSError) as exc:
        print(json.dumps(dict(trace_id=trace_id, module='exploratory', operation=args.command,
            status='denied', reason=str(exc), training_authorized=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
