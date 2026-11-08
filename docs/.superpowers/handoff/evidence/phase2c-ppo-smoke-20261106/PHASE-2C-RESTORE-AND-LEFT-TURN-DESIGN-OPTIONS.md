# Phase 2C Baseline Restore and Left Turn Design Options

Date: 2026-11-06
Target worktree: `D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit`
Source baseline: `6c479540134d98d7d27331879c926c8ba82c2965`

## Baseline blocker resolved

Restored only `scripts/wp3_owner_train_dev_eval.py` from the source baseline. Its SHA-256 is `4abf53213ea2e44c04ebbf53242b1f78eb4f28c35dea2999342321a21f607f68`.

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\Projects\.worktrees\F1-RL\drive-visual-ppo-compat\.venv\Scripts\python.exe' -B -m pytest tests\research\test_wp3_smoke_runtime_profile.py tests\research\test_wp3_eval_only_coordinator.py tests\integration\test_wp3_registered_crash_terminal_contract.py tests\research\test_wp3_owner_train_dev_eval.py -q -p no:cacheprovider
```

Result: `73 passed in 6.57s`. This supersedes only the 73-test blocker recorded in the original Phase 2C report. The original report and Phase 2A evidence remain unchanged. The target still has no Git HEAD (orphan snapshot), so this restoration is not represented as a normal tracked diff.

## Read-only driving diagnosis

Inputs: the 12 Phase 2C per-step JSONL traces and the Phase 2C report under `D:\Projects\.evidence\F1-RL\phase2c-ppo-smoke-20261106\synced-smoke-run\`.

| Track | Observed outcome | Trace evidence | Assessment |
|---|---|---|---|
| Right Turn | 3/3 valid `physical_crash`, all `bounded_road_body`; each stopped before checkpoint 1 | At the terminal steps the cars were still near the end of the entry leg (x about -0.10 to 0.14 m; z about 69.77 to 70.27 m), with route highwater 65.77–66.14 m. In the preceding 10 actions throttle averaged 0.84–1.00, brake stayed 0, and absolute steering reached 1.0. | Strong evidence the policy did not complete the turn before reaching the entry-road end. Physical event is contract-valid. Policy contribution is medium confidence because there is no control counterfactual. |
| Ramp | 2/3 valid `physical_crash` (`wheel_hard_stop`); C-seed0 finished | Both crashes were on `road-incline` at z about 49.16–49.80 m and 8.27–9.10 m/s. The preceding 10 actions averaged full throttle, zero brake, with full steering reached. | Driving/suspension interaction is plausible, medium confidence. The recorded v4 witness lacks wheel identity and deflection, so the exact wheel-limit mechanism is not recoverable from these logs. |
| Slalom | 3/3 valid crashes: 2 `bounded_road_body`, 1 `wheel_hard_stop` | All three kept full throttle and no brake during the final 10 steps, with steering reaching full deflection. Two reached z about 85 m near x=-6 m before crashing; one left the road earlier at x about 6.9 m, z about 54.6 m. | Policy line/speed control is a plausible contributor, medium confidence. The distinct failure locations suggest more than one slalom failure mode. |

These are valid Physics v4 crash terminals, not solver-invalid transitions. The smoke establishes episode outcomes and recorded action/position evidence; it does not establish counterfactual causality or substep equivalence. No Physics, PPO, reward, observation, checkpoint, or TRAIN asset was changed for this diagnosis.

## Left Turn source and roster constraints

- The manifest reserves `left-turn-v1` for `evaluation/left-turn-v1.json`, and the experiment config lists that identity as HELD-OUT. The geometry file is absent from the target snapshot; no HELD-OUT geometry was opened.
- A new identity must be used to avoid colliding with that reserved held-out identity. Proposed name: `left-turn-train-v1`.
- The formal TRAIN roster is exactly four identities in `TRAIN_IDS` and the experiment config. It is consumed across preflight, WP7 protocol/evaluation, feasibility/audits, local demo, and tests. An official fifth TRAIN changes protocol assumptions and future evaluation requirements.
- Existing frozen PPO models were trained against the four-track roster. Adding a fifth TRAIN identity does not retroactively make those models trained on it; the left-turn run would be zero-shot smoke evidence.

## Design options for approval

### A. Add a fifth official TRAIN identity as a mirrored Right Turn (recommended)

Create `left-turn-train-v1` by reflecting the TRAIN Right Turn across x=0, preserving road width, segment lengths, spawn, slow zone, checkpoint dimensions, and route distance. Add the new identity to the official TRAIN manifest/config/roster and update directly affected roster-contract tests. Smoke the existing frozen PPO models on this asset without retraining or full evaluation.

Why: the mirrored geometry isolates turn handedness and gives the requested new TRAIN route. It preserves the existing HELD-OUT identity. Cost: roster and protocol consumers need coordinated updates; existing models retain their four-track provenance.

### B. Create a mirrored TRAIN-folder smoke fixture outside the official roster

Create a separate mirrored asset for engineering smoke only; do not add it to `TRAIN_IDS` or the official experiment config.

Why: smallest scope and cleanly preserves the frozen four-track protocol. Cost: it is not an official TRAIN identity and so does not fully satisfy a formal fifth-TRAIN interpretation.

### C. Add a separately authored fifth official TRAIN Left Turn

Create a left turn with independently chosen geometry, then expand the official roster.

Why: permits a less constrained course design. Cost: geometry, rather than handedness alone, becomes a confound when comparing Left Turn with Right Turn; feasibility and route design require additional choices.

## Proposed success checks after design approval

1. Preserve the held-out manifest entry and never read its geometry.
2. Validate new asset schema, manifest hashes, route/checkpoint direction, and v4 feasibility without changing physics thresholds.
3. Verify no model/checkpoint or existing TRAIN asset hash changes.
4. Smoke both left and right turns with the same deterministic models, reset seed, cap, and v4 runtime provenance.
5. Report terminal class and action/position traces, separating valid crash outcomes from invalid transitions. Do not start training or full evaluation.
## Post-restore integrity check

Recomputed against Phase 2C `integrity_before` and `integrity_after` in `phase2c-results.json`. All selected smoke models and all four existing TRAIN assets match both recorded snapshots.

| Artifact | Current SHA-256 | Phase 2C match |
|---|---|---|
| C-seed0 | `15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492` | yes |
| B-seed1 | `695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021` | yes |
| A-seed2 | `e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc` | yes |
| straight-sprint-v1 | `9efe209780cfa8358f8622d38bb5d56db30425a57dcebebc16e6e447a10b5f00` | yes |
| slalom-v1 | `e9dd588740189f7ed857de8e8b8f5eeb99f3161fdb9b8d3fa05d18978cdefa53` | yes |
| right-turn-v1 | `a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e` | yes |
| ramp-v1 | `6c0da56966962363ed601c8fda2343006c207ee9ea6497deb72e6f6018721bd3` | yes |
