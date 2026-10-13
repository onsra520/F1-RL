# H-01 remediation — canonical C2 protocol version 2

**Status: IMPLEMENTED ON REVIEW BRANCH / NOT VERIFIED BY FULL TEST SUITE / NOT APPROVED FOR SCIENTIFIC FREEZE.**

Review branch: `audit/h01-canonical-c2-contract`, forked from audit baseline `0cc4344` of `recovery/wp7-rebuild`. No change to original legacy v1 YAML, model ZIPs, TRAIN track files, historical manifests, or WP7 recovered evidence.

## Why this change exists

The canonical `ppo-generalization-v1.yml` defines a **legacy v1** observation and reward identity, while real WP7 C2 PPO training used authored surface navigation, route-highwater reward and an explicit off-track penalty. Those are different scientific contracts. Merely setting v1 `status: frozen` would misrepresent model provenance.

## Versioned contract

- `ExperimentDefinition` now interprets `experiment.protocol_version` **before** validating YAML root/semantic sections. Both versions are strict and mutually incompatible, including direct dataclass construction.
- **v1** retains its six-field semantics, `target_progress` reward, the original `ExperimentDefinition` positional constructor, and its existing checked-in YAML; v1 must reject v2-only fields.
- **v2** requires `dynamics`, `navigation`, the C2 route observation identity, `route_progress` reward with high-water semantics, `offtrack` reward penalty and top-level off-track consecutive-step/lateral limit.
- `RewardExperimentConfig` keeps v1 target-progress unchanged; v2 uses separate `route_progress_weight_per_meter` and `offtrack_penalty` fields.
- `OffTrackExperimentConfig` strictly validates positive finite lateral limit and positive integer consecutive-step limit.
- `require_frozen()` handles version-specific required rewards, preserves train/HELD-OUT split validation, requires **at least three paired seeds**, refuses the original pilot/recovery training profile for v2, and continues to refuse any draft experiment.
- `configs/experiments/ppo-generalization-c2-v2.yml` is explicitly **draft**, training **pilot_owned**, with **empty final seed arrays**, same WP1B vehicle/sensors/simulation and WP3 metadata track partition. This is a representation of the recovery candidate, **not an official frozen protocol**.
- `verify_draft_c2_alignment()` compares v2 canonical semantics, original environment/split, reward/off-track constants, and pilot reference against the historical `RecoveryCandidate`. It is read-only; no HELD-OUT environment is constructed.

## Required review gates before integration

1. **Run focused tests**:
   ```powershell
   cd D:\Projects\.worktrees\F1-RL\wp7-rebuild
   # After fetching/checking out the isolated H-01 review branch:
   .\.venv\Scripts\python.exe -m pytest tests/learning/experiments -q
   ```
2. **Run full regressions and compatibility**: `python -m pytest -q`, `python -m compileall -q f1rl scripts`, `uv pip check --python .\.venv\Scripts\python.exe`.
3. **Independent scientific review**: decide whether a point-agent simulation is the intended claim or whether body-aware collision needs a new physics protocol. Review Right Turn route corridor, missing provenance, failed stochastic TRAIN screen, independent evidence backups.
4. **Prospective execution**: add an experiment-owned immutable final training profile, signed-off scientific seed/replicate/evaluation requirements, model/track/source/config fingerprints, provenance capture at run start and an evidence gate for the *new* candidate.
5. **Historical evidence isolation**: the original `wp7_freeze_readiness` checker intentionally rejects committed source changes outside its audited allowlist; this H-01 branch changes `experiments/definition.py`. Do not bypass it by casually adding experiment-source files to `AUDIT_ONLY_PATHS`. Continue to evaluate the two original 200k archives on baseline `recovery/wp7-rebuild` until an independent compatibility review establishes a safe migration.

## Test and scientific limitations

The new tests cover v1/v2 semantic separation, draft/not-frozen constraints, numeric off-track validation, reward keys, missing/extra fields, paired seeds, pilot-profile freeze denial, and canonical↔recovery alignment.

**Test status when writing this record:** the Desktop Commander monthly quota blocked local Windows execution; there was no tracked GitHub Actions CI workflow. This PR must remain **draft** until tests execute and the scientific contract is independently reviewed. Do not claim tests passed merely because code and test files were committed.

No training, HELD-OUT evaluation, weight modification, WP8, or merge to `main` or `recovery/wp7-rebuild` was performed.
