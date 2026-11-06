# F1-RL — Phase 2C: Sync Phase 2B & PPO Smoke

**Execution result: 12/12 smoke cases completed. Acceptance status: BLOCKED** because the exact 73-test command cannot pass in the incomplete target snapshot: one unrelated baseline compile test requires a script that is absent from that snapshot. No source logic outside Phase 2B was added to bypass this.

## Phase 2B sync and target state

The Phase 2B source was `D:\Projects\.worktrees\F1-RL\phase2b-smoke-profile-wiring`, branch `fix/f1rl-phase2b-smoke-profile-wiring`, still at baseline commit `6c479540134d98d7d27331879c926c8ba82c2965` with four uncommitted paths. Those four files were the complete status/diff scope and were copied byte-for-byte into the requested target `D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit`:

- `f1rl/research/wp3_eval_only_coordinator.py`
- `scripts/wp3_eval_only_coordinator.py`
- `tests/research/test_wp3_eval_only_coordinator.py`
- `tests/research/test_wp3_smoke_runtime_profile.py`

The target remains an orphan/unborn snapshot: `git rev-parse --verify HEAD` still fails and no commit was created. The original Phase 2B worktree was not modified by the sync or test run. The target runtime profile resolves to Physics `spatial_road_edge_partial_lifecycle_v4`, race `wp7_directed_lifecycle_crash_contract_v4`, terminal `wp3_physical_crash_terminal_v1`, and `max_abs_tilt_rad=0.7`.

## Tests and integrity

The required full targeted command in the target worktree returned **72 passed, 1 failed**. The failure was `tests/research/test_wp3_owner_train_dev_eval.py::test_cli_compiles`, which tries to read `scripts/wp3_owner_train_dev_eval.py`; that baseline script is absent from the Phase 1 snapshot. It is not a Phase 2B path and was not copied, consistent with the sync scope. Excluding only that path-dependent test gave **72 passed, 1 deselected** in the target.

The same full targeted command in the unchanged Phase 2B worktree returned **73 passed in 6.69s**. All four Phase 2B files in the target have byte-identical SHA-256 to the corresponding Phase 2B worktree files. This verifies the synced profile code, but does not turn the target's one missing-file test into a PASS.

The smoke captured hashes before and after; all were unchanged and match Phase 2A provenance.

Models:

- `C-seed0/model-final.zip`: `15373c0237e883fa2f980dfecbda4809aabc283c0d77a8b4d783e2ab5bf11492`
- `B-seed1/model-final.zip`: `695f1c649235775eb1446ac0718b69b5951a1242b389221a5318db540a716021`
- `A-seed2/model-final.zip`: `e2a0eb0deef34fad7bf2bf2322c0c13854869372263a4652d4c6f1c422d43adc`

TRAIN track assets:

- `straight-sprint-v1.json`: `9efe209780cfa8358f8622d38bb5d56db30425a57dcebebc16e6e447a10b5f00`
- `slalom-v1.json`: `e9dd588740189f7ed857de8e8b8f5eeb99f3161fdb9b8d3fa05d18978cdefa53`
- `right-turn-v1.json`: `a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e`
- `ramp-v1.json`: `6c0da56966962363ed601c8fda2343006c207ee9ea6497deb72e6f6018721bd3`

## Protocol and 12-case results

The run used the Phase 1 source baseline `6c479540134d98d7d27331879c926c8ba82c2965`, registered `spatial_rotating_owner_v3` scenario construction, `wp3_full06_physics_v4` runtime overlay, deterministic CPU PPO inference, reset seed `20261106`, and step limit `2500`. No PPO training, HELD-OUT geometry, or full evaluation ran. **1,275 actions** were recorded across all 12 JSONL episode logs.

| Model | Track | Phase 1 result | Phase 2C result | Steps | Checkpoints | Time (s) | Terminal |
|---|---|---|---|---:|---:|---:|---|
| C-seed0 | Straight | finish, 126 | finish | 126 | 2 | 3.304 | — |
| C-seed0 | Slalom | invalid at attempt 91 | physical crash | 150 | 2 | 6.966 | valid wheel hard stop |
| C-seed0 | Right Turn | invalid at attempt 90 | physical crash | 93 | 1 | 3.114 | valid bounded-road body |
| C-seed0 | Ramp | finish, 199 | finish | 199 | 2 | 7.859 | — |
| B-seed1 | Straight | finish, 108 | finish | 108 | 2 | 3.072 | — |
| B-seed1 | Slalom | invalid at attempt 121 | physical crash | 125 | 2 | 5.658 | valid bounded-road body |
| B-seed1 | Right Turn | invalid at attempt 78 | physical crash | 81 | 1 | 2.753 | valid bounded-road body |
| B-seed1 | Ramp | invalid at attempt 55 | physical crash | 55 | 0 | 2.083 | valid wheel hard stop |
| A-seed2 | Straight | finish, 113 | finish | 113 | 2 | 3.073 | — |
| A-seed2 | Slalom | invalid at attempt 82 | physical crash | 86 | 1 | 4.003 | valid bounded-road body |
| A-seed2 | Right Turn | invalid at attempt 82 | physical crash | 84 | 1 | 2.729 | valid bounded-road body |
| A-seed2 | Ramp | invalid at attempt 55 | physical crash | 55 | 0 | 2.342 | valid wheel hard stop |

Phase 1's `steps` for invalid rows counted accepted steps before the failing attempt; the table shows the failed attempt number. Phase 2C counts its terminal transition as an executed step.

### Outcome comparison

| Phase | Finish | Valid physical crash | Invalid transition | Hard-route violation | Step limit |
|---|---:|---:|---:|---:|---:|
| Phase 1 v3 smoke | 4 | 0 | 8 | 0 recorded | 0 recorded |
| Phase 2B replay (the 8 prior invalid cases) | not replayed | 8 | 0 | 0 | 0 |
| Phase 2C (all 12) | 4 | 8 | 0 | 0 | 0 |

All eight Phase 2C crashes passed `validate_registered_crash_terminal_info`. Their terminal reasons and steps match the Phase 2B v4 counterfactual replay. The four Phase 1 finishes remain finishes with the same step counts and completed checkpoint counts. This is episode-level outcome/step comparison; no transition/substep or full trajectory equivalence is claimed. V4 changes the trajectory from the start.

The finish count stayed **4/12**. Converting eight invalid transitions into valid crash terminals improves runtime/contract reporting; it does **not** demonstrate better PPO driving performance.

## Non-finish analysis

- **Right Turn — 3/3 bounded-road body crashes.** The registered terminal witnesses identify `bounded_road_body` with source `wp3_grounded_applied_road_crash_v4_dev`; terminal contract validation passed. Near the crashes, speeds were about 6.1–7.9 m/s. The final actions had no brake; two cases were still at full throttle, while C-seed0 had reduced throttle to about 0.44. The terminal event classification is high-confidence. PPO speed/line control contributed plausibly, with medium confidence; this smoke has no alternate-policy counterfactual to isolate responsibility.
- **Ramp — 2/3 wheel hard stops.** B-seed1 and A-seed2 ended with registered `wheel_hard_stop` witnesses from `wp3_grounded_wheel_hard_stop_crash_v4_dev`; contract validation passed. The preceding actions kept throttle at 1.0 with no braking, at roughly 9.1 and 8.3 m/s; steering was at or near full deflection. The physical event is high-confidence; the policy contribution is medium-confidence absent a counterfactual. C-seed0 finished Ramp.
- **Slalom — 3/3 valid crashes.** Two were `bounded_road_body`; C-seed0 was `wheel_hard_stop`. These are accepted physical terminal outcomes, not invalid solver transitions.

No evidence from this run indicates a Physics v4 implementation bug. The remaining failures are eight valid physical crashes under the active runtime contract. Policy behavior is a plausible contributing factor, but this smoke alone does not prove that the policy is the sole cause.

## Recommended next phase

1. **High priority — Right Turn driving diagnosis:** review the complete action/position trace around body contact, quantify road-edge margin and speed/steering timing, and compare against a separately approved control counterfactual. Do not infer a need to change collision geometry from these valid contacts alone.
2. **High priority — Ramp driving diagnosis:** examine throttle/brake/steering and suspension witness timing for the two wheel hard stops. Separate policy approach speed from the registered wheel-travel limit before considering any physics change.
3. **Medium priority — Slalom policy behavior:** inspect the three valid crash traces and compare the body-contact and wheel-hard-stop trajectories.
4. Keep PPO/reward/observation changes and retraining for a separately scoped, approved phase. No fix or training was started here.

## Checklist and final status

- **1. Phase 2B located and full diff checked:** PASS.
- **2. Only the four Phase 2B coordinator/CLI/test files transferred:** PASS; files are byte-identical; original Phase 2B worktree unchanged.
- **3. Profile and model/asset hashes verified:** PASS. Full 73-test target run: **BLOCKED** by the missing unrelated baseline script; 72 target tests passed, and 73 passed in the complete Phase 2B source worktree.
- **4. Deterministic 12-case smoke:** PASS, 12/12.
- **5. Phase 1 seeds, step cap, protocol, action logs and terminal outcomes recorded:** PASS.
- **6. Phase 1/Phase 2B comparison and non-finish diagnosis:** PASS, with causality limits stated above.
- **7. Report, replay and logs exported outside the worktree:** PASS.

**Overall: BLOCKED on the strict acceptance gate** because the complete 73-test suite cannot pass in the specified target snapshot without adding a non-Phase-2B baseline script, which was outside the authorized sync scope. No commit, push, merge, Physics/PPO/reward/observation/terminal changes, retraining, HELD-OUT access, or full evaluation occurred. The target remains orphaned.

## Reproducibility artifacts

All Phase 2C artifacts are outside the worktree in `D:\Projects\.evidence\F1-RL\phase2c-ppo-smoke-20261106\synced-smoke-run\`:

- `replay_phase2c.py` — deterministic replay driver using the synced target worktree.
- `phase2c-results.json` — profile provenance, all 12 outcomes, before/after integrity hashes, and comparisons.
- `episodes\<model>__<track>.jsonl` — per-step PPO actions, controls, progress/checkpoints, wheel support, crashes, and terminal telemetry for all 12 cases.
- `PHASE-2C-PPO-SMOKE-REPORT.md` — this report.
