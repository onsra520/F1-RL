"""Generate/check SHA256 evidence manifest for *independently restored* files.

No Google Drive credentials or network operations are included. Creating a
manifest is not proof of an off-device backup or independent restore.

Examples (PowerShell):
  python scripts/wp7_evidence_manifest.py create --source "D:\\archive\\run" --manifest "D:\\archive\\run-manifest.json"
  python scripts/wp7_evidence_manifest.py verify --source "D:\\restore\\run" --manifest "D:\\archive\\run-manifest.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning.evidence_backup import (
    build_evidence_manifest, serialize_manifest, verify_evidence_manifest,
)


def _unique_pairs(pairs):
    doc={}
    for key,value in pairs:
        if key in doc:
            raise ValueError(f"duplicate evidence JSON key: {key}")
        doc[key]=value
    return doc


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("create","verify"))
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    manifest=args.manifest.resolve()
    if args.operation=="create":
        if manifest.is_relative_to(source):
            raise ValueError("manifest output must be outside the scanned evidence tree")
        document=build_evidence_manifest(source)
        # Do not overwrite a previously registered immutable evidence record.
        with manifest.open("x",encoding="utf-8") as handle:
            handle.write(serialize_manifest(document))
        print(f"MANIFEST CREATED: {len(document['entries'])} files")
    else:
        document=json.loads(manifest.read_text(encoding="utf-8"),object_pairs_hook=_unique_pairs)
        count=verify_evidence_manifest(source,document)
        print(f"RESTORE VERIFIED: {count} files, SHA256 exact")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
