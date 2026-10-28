import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research.engineering_attestation import _identity, create_current_attestation, verify_current_attestation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    _identity(root)
    if root != ROOT.resolve():
        raise ValueError('attester must run from the target checkout')
    if args.output.resolve().is_relative_to(root):
        raise ValueError('attestation output must be outside the source tree')
    if args.verify:
        print(json.dumps(verify_current_attestation(root, json.loads(args.output.read_bytes())), sort_keys=True))
    else:
        if args.output.exists():
            raise FileExistsError('attestation output already exists')
        document = create_current_attestation(root)
        with args.output.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': 'CURRENT_ENGINEERING_ATTESTATION_CREATED', 'source_commit': document['source_commit']}, sort_keys=True))


if __name__ == '__main__':
    main()
