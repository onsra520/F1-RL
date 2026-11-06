# F1-RL — Phase 2B: Smoke Harness Runtime Profile Wiring Fix

**Status: PASS** — runtime-profile wiring and fail-fast behavior implemented and validated. Changes remain uncommitted on `fix/f1rl-phase2b-smoke-profile-wiring`; no commit, push, or merge was made.

## Root cause and call path

Phase 1's standalone smoke runner selected registration profile `spatial_rotating_owner_v3`. `load_registration` and `prepare_scenarios` propagated Physics v3 (`spatial_rotating_compliant_lifecycle_v3`) and race v2 (`wp7_directed_lifecycle_owner_v2`); the runner did not apply the FULL-06 runtime overlay or compare requested and effective profiles. This explains the eight v3 solver rejections. Phase 2A reproduced them and showed the same models, TRAIN tracks, and reset seed ending as valid `physical_crash` episodes under the FULL-06 v4 overlay. The evidence identifies a harness selection/wiring bug, not a Physics v4 defect or asset/model drift.

The source-bound coordinator already set v4 versions inside `_production_episode`, but those values were implicit overrides: the CLI accepted no requested runtime profile, the resulting factory was not checked against a named profile, and episode rows did not carry requested/effective provenance. Phase 2B names that runtime bundle and wires it through the coordinator.

Call path after the change:

`CLI --runtime-profile (default wp3_full06_physics_v4)` → `run_engineering(runtime_profile=...)` → `_production_episode(..., runtime_profile)` → v3 registration/evaluation factory → `apply_runtime_profile` → `validate_runtime_profile` → `_evaluate_episode` → environment/Physics v4 + race v4 + registered terminal contract.

The CLI passes the request at [scripts/wp3_eval_only_coordinator.py:20](D:/Projects/.worktrees/F1-RL/phase2b-smoke-profile-wiring/scripts/wp3_eval_only_coordinator.py:20) and [scripts/wp3_eval_only_coordinator.py:54](D:/Projects/.worktrees/F1-RL/phase2b-smoke-profile-wiring/scripts/wp3_eval_only_coordinator.py:54). The FULL-06 mapping, effective-profile comparison, and mismatch exception are at [wp3_eval_only_coordinator.py:26](D:/Projects/.worktrees/F1-RL/phase2b-smoke-profile-wiring/f1rl/research/wp3_eval_only_coordinator.py:26) and [wp3_eval_only_coordinator.py:46](D:/Projects/.worktrees/F1-RL/phase2b-smoke-profile-wiring/f1rl/research/wp3_eval_only_coordinator.py:46). Production applies and validates the profile before evaluator entry at [wp3_eval_only_coordinator.py:201](D:/Projects/.worktrees/F1-RL/phase2b-smoke-profile-wiring/f1rl/research/wp3_eval_only_coordinator.py:201). Requested/effective provenance is attached to episode records; injected evaluator records explicitly say effective configuration was not available.

## Related commits

- `1732fdf51aa8b33e127c0ebba728e6099b090554`: introduced the registered rotating-owner v3 profile.
- `4297bb7e3250c37b51f04eb0194885a5e80e00ae`: integrated Physics v4 and the versioned crash terminal.
- `c057135a329af59e07d9823a7bedf4b2b06f8682`: added finite-tire edge support and certified hard-stop terminal paths.
- `6c479540134d98d7d27331879c926c8ba82c2965`: Phase 2B source baseline; Phase 1 and FULL-06 used this source head.
- No Phase 2B commit exists; `HEAD` remains the baseline commit.

## Changed files and behavior

- `f1rl/research/wp3_eval_only_coordinator.py`: adds the named FULL-06 profile spec, effective-profile resolver/validator, pre-evaluator mismatch failure, and requested/effective provenance. The optional injected-evaluator seam reports `effective: null` with an explicit unverified status when it cannot supply runtime data.
- `scripts/wp3_eval_only_coordinator.py`: accepts `--runtime-profile`, defaults to the named FULL-06 v4 profile, and forwards the request.
- `tests/research/test_wp3_smoke_runtime_profile.py`: covers factory and environment propagation, profile mismatch before evaluator entry, v3 runtime compatibility, registered `physical_crash` terminal validation, and CLI propagation/rejection.
- `tests/research/test_wp3_eval_only_coordinator.py`: checks that an injected evaluator does not claim an unobserved effective profile.

Before Phase 2B, the Phase 1 smoke path labeled the run v4 while running the registered v3/v2 factory. After Phase 2B, the coordinator resolves the requested FULL-06 profile, applies Physics v4 / race v4 / `wp3_physical_crash_terminal_v1` with the FULL-06 tilt envelope (`0.7 rad`), validates the effective factory profile, and records the profile provenance. Unsupported CLI profile names fail argument parsing before a run starts; a factory/profile mismatch raises `RuntimeProfileMismatch` before `_evaluate_episode` is called. The registered v3 factory and environment remain v3/v2 unless explicitly transformed by the separate v4 overlay.

## Validation

Environment: Python 3.12.15, CPU, project virtual environment.

- **PASS** — targeted unit/integration regression command:

  ```powershell
  $env:PYTHONDONTWRITEBYTECODE='1'; & 'D:\Projects\.worktrees\F1-RL\drive-visual-ppo-compat\.venv\Scripts\python.exe' -B -m pytest tests\research\test_wp3_smoke_runtime_profile.py tests\research\test_wp3_eval_only_coordinator.py tests\integration\test_wp3_registered_crash_terminal_contract.py tests\research\test_wp3_owner_train_dev_eval.py -q
  ```

  Result: `73 passed in 7.36s`.

- **PASS** — `git diff --check`.
- **PASS** — bounded deterministic replay of the exact eight Phase 1 model/track/reset-seed cases through the Phase 2B runtime-profile helper and real environment. Result: 8/8 valid registered `physical_crash` terminals, 0 invalid/exception episodes, and all eight terminal steps match the Phase 2A v4 counterfactual.
- **PASS** — terminal telemetry reports race `terminal_reason=physical_crash`, valid research outcome, and a terminal record accepted by `validate_registered_crash_terminal_info`. Reset telemetry reports Physics v4, race v4, and the registered crash contract.
- **PASS** — environment profile remains requested/effective-equal in all eight cases; existing Physics/race/terminal contracts were exercised without changes.
- **PASS** — three frozen model archive hashes and all four TRAIN track asset hashes match the Phase 2A record before and after replay.
- **PASS** — no HELD-OUT geometry opened, no full 96-cell / 5,760-episode evaluation run, and no retraining performed.

| Model | Track | Phase 1 failed step | Phase 2B v4 terminal step | Result |
|---|---|---:|---:|---|
| C-seed0 | Slalom | 91 | 150 | valid `physical_crash` |
| C-seed0 | Right Turn | 90 | 93 | valid `physical_crash` |
| B-seed1 | Slalom | 121 | 125 | valid `physical_crash` |
| B-seed1 | Right Turn | 78 | 81 | valid `physical_crash` |
| B-seed1 | Ramp | 55 | 55 | valid `physical_crash` |
| A-seed2 | Slalom | 82 | 86 | valid `physical_crash` |
| A-seed2 | Right Turn | 82 | 84 | valid `physical_crash` |
| A-seed2 | Ramp | 55 | 55 | valid `physical_crash` |

**Equivalence scope:** episode-level equivalence is established for all eight cases against the Phase 2A v4 counterfactual: same model, track, reset seed, deterministic inference, terminal outcome, and terminal step. Transition/substep-level equivalence is **not** claimed; Physics v4 changes the trajectory from the start, so the v3 failed-attempt substep is not the same state as the v4 crash transition.

### SHA-256 integrity

Frozen models (read-only Phase 1 worktree):

- `C-seed0/model-final.zip`: `15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492`
- `B-seed1/model-final.zip`: `695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021`
- `A-seed2/model-final.zip`: `e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc`

TRAIN track assets in the Phase 2B worktree:

- `straight-sprint-v1.json`: `9efe209780cfa8358f8622d38bb5d56db30425a57dcebebc16e6e447a10b5f00`
- `slalom-v1.json`: `e9dd588740189f7ed857de8e8b8f5eeb99f3161fdb9b8d3fa05d18978cdefa53`
- `right-turn-v1.json`: `a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e`
- `ramp-v1.json`: `6c0da56966962363ed601c8fda2343006c207ee9ea6497deb72e6f6018721bd3`

The model hashes match FULL-06 provenance; the track hashes match Phase 2A's baseline source verification. None of these files changed.

## Evidence and remaining risks

Phase 2B artifacts are in `D:\Projects\.evidence\F1-RL\phase2b-smoke-profile-wiring-20261106\`:

- `eight-case-v4-profile-replay.json` — case results, terminal/reset telemetry, profile provenance, and before/after hashes.
- `replay_phase2b.py` — bounded replay driver.
- `eight-case-v4-profile-replay-summary-bug-attempt.json` — preserved first replay output; its episode results were valid, but an evidence-only serializer counted booleans incorrectly. The corrected driver checks booleans before integers; the final artifact above reports 8/8 valid terminals.
- Phase 2A report and three JSON inputs remain under `D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106\`; none were modified.

The guarded production `run_engineering` CLI campaign was not invoked. Its admission path requires a clean, committed source tree, which conflicts with the requested pre-commit validation boundary; its default external history path is also unavailable in this workspace. The actual deterministic eight-case replay, CLI-to-runner profile propagation tests, production-episode profile test, and real environment terminal tests were run instead. This did not require changing the guard or making a commit.

The changes affect only the smoke coordinator/CLI and tests. No Physics engine, collision geometry, thresholds, PPO, reward, observation, checkpoints, or TRAIN assets changed. The report and all replay evidence are outside the worktree. Code-review checks found no remaining standards or spec gaps.

## Conclusion

**PASS** for Phase 2B acceptance criteria. The smoke runtime profile now matches the requested FULL-06 Physics v4/race v4/terminal contract, mismatch is fail-fast, v3 runtime behavior remains covered, all eight bounded cases terminate validly, and frozen inputs remain unchanged. Work stops here before commit/push/merge, pending user acceptance.
