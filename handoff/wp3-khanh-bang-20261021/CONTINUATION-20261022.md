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
