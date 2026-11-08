# F1-RL — Phase 2C: Physics v4 PPO Smoke Test

**Status: BLOCKED before episode execution.** No PPO episode was run and no source or model file was changed.

## Preflight finding

The requested target `D:\Projects\.worktrees\F1-RL\physics-v4-phase1-audit` is an unborn snapshot worktree: `git status` reports `No commits yet on baseline/physics-v4-phase1-audit`, and `git rev-parse --verify HEAD` fails. The Phase 2B change has not been applied there. Its implementation remains uncommitted on a separate branch/worktree:

- Target: `baseline/physics-v4-phase1-audit`
- Phase 2B implementation: `fix/f1rl-phase2b-smoke-profile-wiring`
- Both worktrees have source HEAD `6c479540134d98d7d27331879c926c8ba82c2965`; the Phase 2B changes are still working-tree modifications, not a commit.

The Phase 1 snapshot contains an older source coordinator with a hardcoded `ground_motion_version=PHYSICS` overlay, but it does **not** define `wp3_full06_physics_v4`, the Phase 2B requested/effective validator, or the Phase 2B CLI script. `scripts/wp3_eval_only_coordinator.py` is absent from the target worktree. Therefore checklist item 1 (“confirm Phase 2B has been applied and `wp3_full06_physics_v4` works”) cannot pass in the specified worktree.

The Phase 2B implementation cannot be ported into the target under this phase’s “no code edits” constraint. Running episodes from the Phase 1 snapshot would not verify the requested Phase 2B runtime-profile wiring, so execution stopped before model loading or PPO inference.

## Inputs and previously recorded results

Phase 1 evidence confirms the exact requested protocol: source `6c479540134d98d7d27331879c926c8ba82c2965`, deterministic inference, reset seed `20261106`, 2,500-step cap, models `C-seed0`, `B-seed1`, `A-seed2`, and all four TRAIN tracks. Phase 1 recorded 4 finishes and 8 `invalid_physical_transition` episodes.

| Model | Track | Phase 1 result / steps | Phase 2B v4 replay | Phase 2C |
|---|---|---|---|---|
| C-seed0 | Straight | finish / 126 | not replayed | not run — blocked |
| C-seed0 | Slalom | invalid / 90 accepted steps | valid `physical_crash` / 150 | not run — blocked |
| C-seed0 | Right Turn | invalid / 89 accepted steps | valid `physical_crash` / 93 | not run — blocked |
| C-seed0 | Ramp | finish / 199 | not replayed | not run — blocked |
| B-seed1 | Straight | finish / 108 | not replayed | not run — blocked |
| B-seed1 | Slalom | invalid / 120 accepted steps | valid `physical_crash` / 125 | not run — blocked |
| B-seed1 | Right Turn | invalid / 77 accepted steps | valid `physical_crash` / 81 | not run — blocked |
| B-seed1 | Ramp | invalid / 54 accepted steps | valid `physical_crash` / 55 | not run — blocked |
| A-seed2 | Straight | finish / 113 | not replayed | not run — blocked |
| A-seed2 | Slalom | invalid / 81 accepted steps | valid `physical_crash` / 86 | not run — blocked |
| A-seed2 | Right Turn | invalid / 81 accepted steps | valid `physical_crash` / 84 | not run — blocked |
| A-seed2 | Ramp | invalid / 54 accepted steps | valid `physical_crash` / 55 | not run — blocked |

Phase 2B replayed the eight Phase 1 invalid cases only. It recorded five `bounded_road_body` crashes and three `wheel_hard_stop` crashes, all accepted by the registered terminal contract. It did not replay the four Phase 1 finishes and is not a 12-case Phase 2C evaluation.

## Failure analysis limits

Phase 2A attributed the six Slalom/Right Turn v3 exceptions to the v3 full-road-coverage rejection and the two Ramp exceptions to bump-limit rejection. These were expected rejections under the active v3 contract; they do not establish improved PPO driving under v4. Phase 2B’s eight valid crashes likewise establish valid terminal handling, not a PPO performance improvement. Since Phase 2C did not execute, there are no Phase 2C action traces, elapsed times, checkpoint counts, finish/crash/step-limit statistics, or new Right Turn/Ramp driving conclusions.

## Checklist

- **1. Phase 2B profile applied and active in target:** BLOCKED — named profile/CLI wiring absent from target worktree.
- **2–5. Deterministic 12-case PPO run and action/terminal logging:** NOT RUN — stopped before episode execution.
- **6. Phase 1 → Phase 2B → Phase 2C comparison:** PARTIAL — prior Phase 1 and eight-case Phase 2B evidence summarized; Phase 2C unavailable.
- **7. PPO failure analysis for this smoke:** NOT RUN — would require Phase 2C traces.
- No code, Physics, reward, observation, model, or asset changes; no retraining, full evaluation, HELD-OUT access, commit, push, or merge.

## Evidence paths

- Phase 2C preflight: `D:\Projects\.evidence\F1-RL\phase2c-ppo-smoke-20261106\preflight.json`
- Phase 1 results: `D:\Projects\.evidence\F1-RL\phase1-physics-v4-audit-20261106\smoke-summary.json` and `smoke-episodes\*.jsonl`
- Phase 2A diagnosis: `D:\Projects\.evidence\F1-RL\phase2a-root-cause-audit-20261106\report.md`
- Phase 2B eight-case replay: `D:\Projects\.evidence\F1-RL\phase2b-smoke-profile-wiring-20261106\eight-case-v4-profile-replay.json`

**Conclusion: BLOCKED.** Continue only after the Phase 2B wiring is available in the authorized target worktree or the Phase 2C target is explicitly changed. No training or fix work was started.
