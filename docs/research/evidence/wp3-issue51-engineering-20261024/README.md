# WP3 Issue #51 — New, Unregistered Eval-only Engineering Evidence

This folder is a **new** read-only-on-input engineering evaluation output produced by
the standalone coordinator on **source commit `0886a68dd1a8e17c0592f2399380665fef3d6734`**.

- Run: `WP3-ENG-ISSUE51-20261024-REPLAY2`
- 16/16 sample episodes completed without runtime invalids; 8 finishes, 5 physical crashes,
  1 hard-route violation, 2 step limits.
- Two frozen PPO checkpoints `A-seed1` and `C-seed0`, four original TRAIN tracks,
  two strata and original paired environment/policy seed slot **episode index 4**.
- Engineering profile: `spatial_road_edge_partial_lifecycle_v4`;
  Crash Terminal v1, `wp7_directed_lifecycle_crash_contract_v4`.
- 12/12 original final checkpoint SHA-256 verified, original 96-cell / 5,760-slot
  roster verified. All original checkpoint, roster and provenance hashes unchanged.
- The 16 cell JSONLs contain append-only started/completed-or-invalid records,
  with individual SHA-256 digests in `WP3-ENGINEERING-EVALUATION.json`.
- No retraining, no HELD-OUT and no modification of historical Issue #51 Eval evidence.
- **This is NOT the 5,760-episode registered evaluation. No independent Physics v4
  signoff and no official candidate ranking are claimed. The full gate remains BLOCKED.**

The engineering result belongs to source HEAD above; this **evidence branch** has
a different commit for packaging and does not change that execution source.
