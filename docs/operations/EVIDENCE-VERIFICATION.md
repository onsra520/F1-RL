# Evidence verification commands

The commands below verify recorded file bytes or report historical source provenance. They **do not** issue new scientific admission, attest an independent off-device backup, run PPO training, or authorize HELD-OUT.

## Stable workflow names

| Versionless command | Compatibility script | Source implementation |
| --- | --- | --- |
| `evidence evidence-manifest` | `scripts/wp7_evidence_manifest.py` | `src/validation/evidence_manifest_workflow.py` |
| `evidence verify-run` | `scripts/wp7_verify_run_evidence.py` | `src/validation/run_evidence_workflow.py` |
| `evidence historical-source-audit` | `scripts/wp7_historical_source_audit.py` | `src/validation/historical_source_audit_workflow.py` |
| `evidence verify-frozen-assets` | `scripts/verify_consolidated_wp3_assets.py` | `src/validation/frozen_asset_integrity.py` |
| `evidence verify-historical-records` | `scripts/verify_historical_source_records.py` | Existing validation API and historical wrapper |

## Suggested read-only workflow

With a Python environment containing the declared extras, from the repository root:

```powershell
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows evidence evidence-manifest --help
python -m scripts.workflows evidence verify-run --help
python -m scripts.workflows evidence historical-source-audit --help
```

The first command verifies bundled frozen checkpoint/report/cell bytes. `--help` for the others does not create an evidence record or touch model data.

For a newly copied, **non-frozen** artifact directory, create a SHA-256 manifest *outside* the source tree and verify a genuinely independent copy:

```powershell
python -m scripts.workflows evidence evidence-manifest create --source <original-archive> --manifest <new-external-manifest.json>
python -m scripts.workflows evidence evidence-manifest verify --source <restored-copy> --manifest <new-external-manifest.json>
```

The manifest must be new; overwrite is rejected. Verification rejects tampered bytes, extra/missing files, duplicate JSON keys, unsafe relative paths and symlinks. Self-contained hashes verify consistency only: external signatures and separately retained evidence are needed for authenticity and independent backup claims.

The historical source auditor writes an explicit *new report outside the repository* and leaves `scientific_acceptance_permitted`, `training_permitted`, `heldout_permitted` and `training_runtime_provenance_verified` false. It checks historical source against recorded commits, not the current Python environment.

## Provenance boundary

Existing `wp3`/`wp7` and dated filenames inside `docs/research/`, its `evidence/` subtree, archived manifest records and registered IDs are historical identities. This refactor changes only active executable implementations and current documentation; it never rewrites the historical records.

For restoration of locked archives, see [Historical recovery](HISTORICAL-RECOVERY.md); for current source validation, see [Historical provenance](../architecture/HISTORICAL-PROVENANCE.md).
