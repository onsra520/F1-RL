"""Verify independently retained run bytes and optional trusted signatures.

No host runtime recapture, model loading, training or authorization occurs.
"""
import argparse
import json
from pathlib import Path

from src.validation.wp7_run_manifest import read_json, verify_run_manifest
from src.validation.wp7_trusted_reviews import verify_run_reviews

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True, help='digest retained independently of restored JSON')
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--trust-registry', type=Path, help='separately authenticated owner-controlled registry')
    args = parser.parse_args(argv)
    if bool(args.review) != bool(args.trust_registry):
        parser.error('--review and --trust-registry must be supplied together')
    document = read_json(args.manifest)
    result = verify_run_manifest(document, args.artifacts, expected_sha256=args.expected_sha256)
    if args.review:
        result['review'] = verify_run_reviews(read_json(args.review), read_json(args.trust_registry),
            protocol_sha256=document['protocol_sha256'], run_manifest_sha256=document['sha256'])
    print(json.dumps(result, indent=2))
    return 0
