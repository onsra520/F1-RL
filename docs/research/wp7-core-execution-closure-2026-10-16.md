# WP7 Core Execution Closure — engineering increment

Parent: Draft PR #21 (`461388e38e024483553c762a7bfcf28411e24f8e`).
Scope: Python core only. No API, WebSocket, React or Three.js.

## Delivered

1. **Policy-bound training configuration:** `wp7_execution.build_policy_execution(root, policy_id)` derives PPO, sampling, and perturbation seeds and the exact TRAIN roster from one registered policy. `PolicyBoundTrainingConfig.validate` independently reconstructs identity and refuses protocol drift, mismatched config, roster, physics, reset and receipt. `TrainingRunner.run_registered` validates **before** model/artifact creation, then still enforces scientific training denial. Historical C2 runner entrypoints are unchanged.

2. **Seeded stochastic cell execution:** `evaluate_episode(reset_seed=..., deterministic=False)` is an opt-in extension; the historical default remains unseeded/deterministic. Independent environment/policy seed derivation is centralized in `wp7_protocol.py`. `execute_software_fixture_cell` runs actual Gym/evaluator software fixtures with a fresh externally seeded policy stream, model byte fingerprint, structured per-episode metrics, and fail-closed invalid-cell abort. `save_cell_evidence` writes hash-bound JSON without overwriting evidence. **Not a real PPO scientific evaluation.** `evaluate_registered_cell` checks research admission and explicitly refuses execution while the independent trusted PPO stochastic worker and approval are absent.

3. **Traceable core readiness:** `configs/research/wp7-core-readiness-matrix-v1.json` defines nine non-web subsystems, explicit weighted capability denominators, P0 cases, source/test evidence and compatibility requirements. `wp7_readiness.evaluate_readiness` distinguishes documented implementation from **executed test evidence**; a reported percentage never substitutes for critical gate coverage. The `--verify` CLI runs only existing software tests, and keeps science/compatibility unapproved. Neither a source file nor a passing unit test can self-approve G1–G6.

## Scientific and compatibility boundaries

- Registered scientific training, WP7 evaluation, HELD-OUT, freeze and WP8 are still blocked.
- No uncalibrated physical model is declared validated; slalom/right-turn/ramp, airborne race/reset, hard-route failure, empirical calibration and independent review still need work.
- Cell policy RNG ownership is injected via a fresh `policy_for_seed` callback for synthetic software tests; a real isolated Stable-Baselines3/PyTorch stochastic worker must be reviewed before scientific execution.
- `prepare_scenarios(..., seed_index=...)` from PR #21 requires an explicit keyword argument (version/migration note for external callers).
- The remaining compatibility work is not guaranteed to equal exactly 10% of engineering hours: Windows/Linux, deterministic replay, C2 history, and serialization must all be verified.
- An individual subsystem is **not ≥90% accepted** while any P0 remains open, regardless of weighted score.

## Verification and limitations

Run:

```powershell
python -m pytest tests/research/test_wp7_execution_binding.py tests/learning/evaluation/test_stochastic_cell.py tests/research/test_wp7_core_readiness.py -q
python -m pytest -q
python -m compileall -q f1rl scripts
python scripts/wp7_core_readiness.py
```

The readiness CLI intentionally exits with 2 while the acceptance gate is incomplete. It is not a failed software test. Cross-OS exact-HEAD CI is authoritative for regression portability, not scientific authorization.

**Next scoped milestone:** actual independently approved lifecycle race/reset domain, full TRAIN path feasibility, registered hard-route terminal failure contract and isolated PPO stochastic worker, followed by signed admission and empirical evidence; do not claim a 90% complete research product from this integration increment alone.
