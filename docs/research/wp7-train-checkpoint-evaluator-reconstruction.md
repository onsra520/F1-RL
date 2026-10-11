# WP7 recovery — TRAIN-only checkpoint evaluator

**Reconstructed:** 2026-10-11. **State:** mock-verified, NOT a scientific pilot execution.

The new `f1rl.learning.train_checkpoint.evaluate_training_checkpoint()` validates a supplied checkpoint path and hashes its model archive, loads the explicit non-frozen C2 recovery candidate and 200k profile, then evaluates **exactly four TRAIN tracks** through the existing deterministic evaluator and immutable `EpisodeMetrics`/`TrackEvaluationSummary` contracts.

- All track selection comes from `build_training_scenarios()` and must match `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1` in that order.
- The function never constructs or evaluates a HELD-OUT environment.
- A real model archive, if explicitly provided by a future run, would load through `PPOModel.load(device="cpu")`. No model was loaded in this reconstruction session.
- A separately injected `test_policy` is supported **solely for mocks**. Such reports are labeled `injected_test_policy_not_scientific_evidence`.
- Every report has `scientific_evidence: false` until full training/run provenance, checksum-backed artifacts, stable checkpoints, seed independence and registration gates are reviewed.
- Fingerprints of the 200k profile, recovery C2 candidate, original experiment, manifest and four TRAIN assets are included; `checkpoint_sha256` binds the supplied archive bytes.
- Any missing model, invalid episode count, or mismatched evaluation track ID fails closed.

**Not yet a complete pilot executor:** no PPO training has run, no 25k checkpoint evaluation schedule has been implemented against an actual training process, no new 200k confirmation model exists, and original successful WP7 models are irrecoverable. WIN32 `state.json.tmp` rename reliability and durable artifact backup still need separate validation. The right-turn/slalom route-feasibility blockers from preflight remain open; no WP8 or HELD-OUT learned outcomes were accessed.

Focused tests: `tests/learning/evaluation/test_train_checkpoint_recovery.py` with fake archive and mocked policy/evaluator. Final regression totals recorded in subsequent verification.
