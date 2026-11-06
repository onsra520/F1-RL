"""Preparation is permitted; execution requires a separately signed owner grant."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preflight = commands.add_parser('preflight', help='inspect source/config; no Gym reset or PPO model')
    preflight.add_argument('--output-directory', type=Path, required=True)
    preflight.add_argument('--permit', type=Path, required=True)
    for command in ('execute', '_worker'):
        item = commands.add_parser(command)
        item.add_argument('--permit', type=Path, required=True)
        item.add_argument('--authorization', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    from .permit import prepare, read_json
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
            print(json.dumps(dict(permit_sha256=permit['sha256'], state=permit['state'],
                exploratory_execution_authorized=False, training_authorized=False)))
            return 0
        from .pilot import supervise, worker
        permit, envelope = read_json(args.permit), read_json(args.authorization)
        return (supervise if args.command == 'execute' else worker)(root, permit, envelope)
    except (ValueError, OSError) as exc:
        print(json.dumps(dict(status='denied', reason=str(exc), training_authorized=False)))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
