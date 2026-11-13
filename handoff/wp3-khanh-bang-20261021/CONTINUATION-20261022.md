# WP3 CPU PPO continuation — 12/12 TRAIN runs (2026-10-22)

**Training source:** `research/wp3-ppo-tuning-v2` at frozen commit
`fe83f517479f3b990028d6409304fae83b8199a9`.
The source branch has **not** been modified for this artifact publication.

**Evidence branch:** `handoff/wp3-khanh-bang-20261021`.
This commit extends the original 3-model handoff with all 9 subsequent model checkpoints and validation metadata.

## TRAIN completion

- Original 3 completed runs: `C-seed0`, `A-seed0`, `B-seed0`. Original Git artifacts were left untouched.
- Additional 9 completed runs: `D-seed0`, `A-seed1`, `B-seed1`, `C-seed1`, `D-seed1`, `A-seed2`, `B-seed2`, `C-seed2`, `D-seed2`.
- **12/12** runs completed, **65,536 PPO TRAIN steps per run**, **192 periodic checkpoints**, **12 final models**.
- All 12 passed the frozen `ENGINEERING_PASS` full validation; this is **training integrity**, not driving performance or scientific acceptance.

## Artifacts and verification

The `models/<run-id>/` directories contain the nine additional final model ZIPs,
16 periodic checkpoint ZIPs per run, run and telemetry manifests, training-integrity
records, engineering validation, and runtime/state metadata.
`CONTINUATION-20261022-SHA256.json` lists each added artifact's exact SHA-256,
and the separately verified original telemetry SHA-256 and byte length for all nine runs.

**The large original `training-telemetry.jsonl` records are not committed to Git.**
They must accompany the models as portable evidence to re-run full validation.
The original 3-run prerelease asset `WP3-attempt-v8-full-evidence.zip`
remains the baseline; do not confuse these committed model artifacts with the
entire scientific or campaign evidence.

## Limits / next actions

The external historical smoke/preflight evidence at `D:\Projects\.evidence\F1-RL`
was not provided in the initial handoff. Therefore, the immutable campaign
`campaign-state.json` still reflects the prior partial orchestrator state.
A separate local continuation coordinator validated all 12 actual run directories
without rewriting the frozen campaign state. Reconcile that original state only
after retrieving the genuine original prerequisites; **do not synthesize smoke
evidence or edit the locked source/configuration to bypass it.**

The planned TRAIN-only evaluation (5,760 episodes) and scientific WP7/HELD-OUT
admission are **not** complete or authorized by this checkpoint publication.
No HELD-OUT data was accessed.

## Portable telemetry archived on private GitHub Release

- Release: https://github.com/onsra520/F1-RL/releases/tag/wp3-khanh-bang-20261021
- Continuation asset: `WP3-attempt-v8-continuation-9-telemetry-20261022.zip`
- Asset URL: https://github.com/onsra520/F1-RL/releases/download/wp3-khanh-bang-20261021/WP3-attempt-v8-continuation-9-telemetry-20261022.zip
- Size: 250,542,291 bytes
- SHA-256: `726573b374f3aec727457f2c7b7b1f6533ae09079e257577b85ced17fac18926`
- Contents: 9 original `runs/<model>/training-telemetry.jsonl` records and corresponding telemetry manifests, plus the continuation SHA-256 manifest.

For full portable reconstruction, first unpack the original 3-model release asset.
Unpack the continuation telemetry archive into its existing `attempt-v8` directory.
Copy the nine new checkpoint/metadata directories from this handoff branch's
`models/<model>/` to matching `attempt-v8/runs/<model>/` directories.
Verify file digests from `CONTINUATION-20261022-SHA256.json` before any evaluation.
The frozen original `campaign-state.json` remains unreconciled and must not be represented
as a 12/12 authoritative campaign state.