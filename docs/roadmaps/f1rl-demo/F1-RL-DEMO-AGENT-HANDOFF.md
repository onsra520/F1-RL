# F1-RL — Agent Handoff / Resume Contract

> **Document version:** 2.8 (2026-11-11). **Current state:** `M0_PASS; M1_PASS; M2_PASS; M3_PASS_2_WORKERS; M4_ENGINEERING_PASS; M5_CYCLE1_DIAG_COMPLETE_CANDIDATE_NOT_ACCEPTED; OVERALL_NOT_ACCEPTED`.
> This is an actionable **handoff**, not a certification of PPO driving.  
> **Authoritative goals:** [Master Roadmap](F1-RL-DEMO-COMPLETION-ROADMAP.md) · **Proof/limits:** [Evidence Ledger](F1-RL-DEMO-EVIDENCE-LEDGER.md).

## 0. Live takeover snapshot — M5 Cycle 1 ablation (2026-11-11)

- PR #60 and PR #61 **MERGED**. Local `main` and `origin/main` both at `ab8951a9c68f009d7b2a14127b13c07426a2460a`.
- Working branch/worktree: `feat/f1rl-demo-completion` / `D:\Projects\.worktrees\F1-RL\f1rl-demo-completion`; Cycle 1 evidence source HEAD `e3b7f749186d676cbd1a63796e68d0930866cbfe`, source-tree SHA `8030081db7fedd253d306cc6a31c8a365aa3d56f`. Cycle 1 scripts are committed; result, limitations and artifact hashes are recorded in sections 0 and 15 and the linked Evidence Ledger.
- M0 **PASS**, M1 engineering **PASS / READY_FOR_M2**, M2 **PASS**, M3 **PASS for the matched 1-worker/2-worker benchmark only**, and M4 **ENGINEERING_PASS** for its bounded two-model development plan. Ramp V1 stochastic rose from 1/10 to 10/10 under std×0.5. The same candidate scored Slalom 9/10 versus the historical 10/10; std×0.65 scored Slalom 6/10. M5 remains `CANDIDATE_NOT_ACCEPTED`; no PPO fine-tuning or checkpoint freeze. V5 remains custom Auto Drive only; HELD-OUT remains sealed; no main merge.
- Historical real smoke source `6452517e3e887ee83f9d98b3323e3e5089d4c187`; artifact hashes and telemetry independently match. Historical M3 source `8582d9d08826074dbd2b6d32609d9d58639f771d`; its profile SHA `1b9f…` does not match the benchmark profile at that commit, so old timing results do not admit worker count.
- Historical runtime snapshot before M2–M5: **1,365.796 / 86,400 seconds**, provisional remaining **85,034.204 seconds**. This is not the current cumulative total; see Evidence Ledger §0.4 for the later campaign accounting. Cycle 1 used a non-training ablation; no training round was run.
- On `main` `ab8951a`, independent scoped regression: **121 passed, 2 skipped**. Five historical checkpoints loaded, hashes matched and predicted finite actions. Post-merge new real smoke from `6452517` PASS 256 steps / 10 optimizer updates / checkpoint `09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf`; 90.718 seconds and preserved under `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110`.
- **Perf observation:** new smoke rollout 85.255 seconds vs historical 18.040 seconds; cause unknown, avoid speedup/ETA claim and investigate in M0/M3 as needed.
- Any historical claims below about unmerged docs/parallel branches are **superseded by this dated snapshot**. No scientific HELD-OUT access.

## 1. Current operator-readable status

| Key | Status / fact |
|---|---|
| Overall | `NOT_ACCEPTED` — Ramp ablation improved, but Slalom regressed against its fixed-roster baseline; M6+ and final acceptance remain pending |
| Next milestone | **Review Cycle 1 evidence and the Slalom trade-off**; no additional training cycle was opened |
| M0-M10 | M0 PASS; M1 PASS; M2 PASS; M3 PASS (2-worker scope); M4 ENGINEERING_PASS; M5 Cycle 1 diagnosis complete / candidate NOT_ACCEPTED; M6-M10 PENDING |
| Operating permission | Autonomous within owner-approved scope and 24h active workload cap; never merge main |
| Scientific HELD-OUT | **SEALED**; WP7/Gate 2 **BLOCKED** |
| Final model freeze | Not created |
| Three custom acceptance tracks | Not created — must only be generated **after** model freeze |
| 70-run Pure PPO evaluation | Not performed |
| Live real-time 1×/30FPS/p95≤150ms | Not measured |
| Training budget epoch | See Evidence Ledger §0.4 for M2–M5 campaign accounting; the 1,371.220 s figure in older handoff snapshots predates those runs |
| Agent improvement cycles | Cycle 1 non-training ablation complete; no PPO fine-tuning or additional training cycle |
| Remaining gates | Review stochastic Slalom regression and candidate acceptance; later M6+ gates remain pending; V5 remains custom Auto Drive only; seven external WP7 engineering nodes still lack witnesses |

**Do not take the historical 63+3 tests as proof of real PPO driving.** They were scoped code/mock integration tests. Real training must satisfy M2–M4 gates with new evidence.

## 2. Verified repository context as of handoff

- Repository: **`onsra520/F1-RL`** (private GitHub); remote default branch `main`.
- Remote `main` checked after both merges (2026-11-10): `ab8951a9c68f009d7b2a14127b13c07426a2460a`.
- PR #60 documentation has been merged into `main`; all three files are present in this worktree.
- PR #61 integrated parallel supervisor through `d2ee0bc77cfab94e05968f19c7b9cdaef6bb457a` into merged main. The original real PPO artifacts are bound to earlier `8582d9d08826074dbd2b6d32609d9d58639f771d` and remain immutable.
- Known Windows repo: `D:\Projects\F1-RL`.
- Known candidate parallel worktree: `D:\Projects\.worktrees\F1-RL\wp3-parallel-training-campaign`.
- External evidence/artifact storage: `D:\Projects\.archives\F1-RL\demo-completion\`.
- Known Python environment from last scoped audit: `D:\Projects\.venvs\f1rl-wp3-parallel-training\Scripts\python.exe` (check current existence and environment before invoking).
- Frozen historical config: `configs/research/wp3-ppo-tuning-campaign-v3.json`, SHA-256 `8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4`. Original v3 training: 12 models, 65,536 timesteps each; **full C0 first**.
- Historical smoke config `configs/research/local-demo-ppo-smoke-v1.json` describes C-seed0, 256 timesteps, CPU n_envs=1; re-verify CLI/profile source.
- Earlier tests on local parallel branch: **63 targeted PASS + 3 evaluation/report PASS** on Windows, no real PPO training. Docs creation has not rerun any local tests.

**Integration note:** The earlier local-only branch discrepancy was resolved by merging PR #61. Source admission and rollout correctness for later phases must still be checked against their specific worktrees/commits.

## 3. Required immediate M0 reconnaissance

Execute these **read-only discovery checks first** on the Windows PC; adjust paths if host changed:

```powershell
$repo = 'D:\Projects\F1-RL'
git -C $repo fetch origin --prune
git -C $repo status --short --branch
git -C $repo rev-parse HEAD
git -C $repo rev-parse origin/main
git -C $repo worktree list --porcelain
git -C $repo branch -avv
git -C $repo log -6 --oneline --decorate
git -C $repo status --porcelain
Get-FileHash "$repo\configs\research\wp3-ppo-tuning-campaign-v3.json" -Algorithm SHA256
Get-PSDrive -Name D
```

**Fetching** remote refs does not merge; it is authorized reconnaissance. Do not modify uncommitted work in any other worktree. If `docs/f1rl-demo-master-roadmap-20261110` is only on origin, read canonical documents with `git show origin/docs/f1rl-demo-master-roadmap-20261110:docs/roadmaps/f1rl-demo/<file>` (replace `<file>`) or create a new safe docs worktree. **Never force checkout over a dirty tree.**

Then:

1. Verify current local source versus `main` and the 4c8808f parallel candidate. Confirm no pending changes or newer patch. Inspect CLI help and tests; inspect simulator and Track Builder UI contract.
2. Identify actual source branch/worktree for M1–M5. A training branch must be **clean and committed**, not `main`, and pass LOCAL_DEMO source admission. Keep documentation branch separate if convenient; coordinate updates through commits/PRs without silently overwriting another agent's work.
3. Read current `scripts` and training API, not guessed commands. In particular, **`scripts/wp3_campaign.py parallel-run --workers 2` runs the full campaign (C0 first), not a 4,096-step smoke/benchmark.** Short PPO benchmarking requires separately versioned workload support.
4. Re-run scoped non-training regression, including sample real browser frontend connection health and physics/track correctness checks if available; classify missing external WP3 evaluation evidence as `ENVIRONMENT/BLOCKED_EXTERNAL_EVIDENCE` only when independently proven.
5. Verify RAM, CPU/thread count, disk and archives. Confirm monitoring/proc-tree termination and no orphan training processes before starting anything.
6. Determine if the owner-approved **24-hour budget epoch** starts with this new run or whether active measured attempts already exist. Do **not** set cumulative elapsed=0 blindly.
7. Update [Evidence Ledger](F1-RL-DEMO-EVIDENCE-LEDGER.md) with source/host/config and real measured/verified facts. Mark M0 PASS only when its prerequisites hold.

## 4. Owner policy condensed — never reinterpret

**Goal:** Live Pure PPO demo with web state changing as inference+Physics advances, not a recorded/replayed episode. Smooth real-time simulation 1×, frontend ≥30 FPS, p95 backend-to-visible-update ≤150ms, actual browser measurement.

**Finish:** ≥8/10 valid successes **for each** of the four TRAIN tracks and three fresh post-freeze full-feature custom tracks. A missing left-turn training track must be addressed in development (not by secretly altering final acceptance tracks). Full-feature track-suite coverage: left/right/S turns, cone/barrier, ramp, bridge, intersection and elevation.

**Training:** autonomous gate-driven, cumulative 24h active machine workload time, ≤2 improvement rounds, start 2 concurrent PPO jobs; ≤4 GiB RSS/job and ≥8 GiB available RAM reserve; 4 workers only if real benchmark demonstrates benefit and safety. Can change PPO/reward/curriculum/obs/action/physics with documented contract impact + compatibility regression; selective retrain. Every change affecting checkpoint semantics invalidates incompatible evidence.

**Separation:** TRAIN and development tracks allowed for iteration; final custom acceptance tracks do **not exist** until model freeze and are never training feedback. Registered scientific HELD-OUT stays SEALED. Auto Drive/Hybrid runs are not acceptable as PPO success.

**Git:** branch/worktree isolation; agent may commit/push/open PR; **owner alone authorizes merge to main**. Do not approve scientific admission by demo completion. Heavy artifacts outside Git.

**When budget or gate fails:** stop training, preserve best development-selected checkpoint and evidence, report `NOT_ACCEPTED` with exactly which per-track/runtime gate failed. Do not fabricate successful percentages or salvage by retrying selected final evaluation episodes.

## 5. Specific implementation sequence after M0

Do not schedule the entire workflow as one opaque process. For each milestone prepare: (a) immutable plan, (b) source identity, (c) bounded job command, (d) watchdog, (e) raw evidence root, (f) gate decision, (g) updated docs.

1. **M1 Track/physics:** reproduce builder routes and cover left-turn plus bridge/intersection/elevation. Version TRAIN/dev distributions separate from custom final eval. Check source/model compatibility after any physics/obs/action change.
2. **M2 Real smoke:** authorized **actual** C-seed0 256-timestep PPO.learn smoke; verify learning and optimizer/weight/checkpoint/telemetry integrity. Use the real profile. No full campaign here.
3. **M3 Benchmark:** versioned engineering-only 4,096-step jobs (proposed), sequential A/B vs parallel A/B (same seeds/contract) with real PPO learners; benchmark 4 concurrency only if measured benefit, resource headroom and explicit versioned thread policy. Do not mark 4k as v3 C0 PASS.
4. **M4 Train:** choose budget-feasible profile; if using historical v3, run **full-length C0 gate** 65,536 steps before remaining 11. If full-feature skill coverage needs different curriculum, create a new versioned development training protocol. Preserve all fails and independent attempt directories.
5. **M5 Iterate:** use development-only evidence, maximum two improvements, select best compatible checkpoint by declared criteria.
6. **M6 Freeze model**, all relevant source, Physics, observation and action semantics. Write immutable MODEL-FREEZE manifest.
7. **M7 Generate unseen custom tracks** only after freeze, validate geometry *without PPO probing*, pre-register 10 attempts each.
8. **M8 Pure PPO acceptance:** all 70 attempts, ≥8/10 on every row, original failures intact. No acceptance-data-driven repair.
9. **M9 Live web:** click Run, step PPO inference+physics and stream state, test browser live 10-minute session, FPS/1×/p95 and authoritative telemetry causality. No recording/replay.
10. **M10 Package:** replayable **procedure** (not recorded trajectory), immutable evidence, updated docs, PR; independent audit. Owner manually approves merge.

Note: “replayable procedure” means reproducible launch/test steps using genuine policy inference, **never** a recorded trajectory passed off as live.

## 6. Recovery after agent/session interruption

1. Read roadmap, ledger, handoff in that order. Check remote/local SHA and archive hash independently.
2. Identify any active Windows child PIDs and their process creation timestamps (not PID alone). Do not start duplicate job if an earlier child is alive.
3. Compare current clean source/config/model/track contract hashes to those of the last validated attempt.
4. Resume **completed** jobs via validation and artifact inventory only. Training continuation from a partial checkpoint needs *all* optimizer, RNG, environment/reset/sampler and counter states. If unknown/incomplete, preserve old attempt and create a newly authorized attempt, never pretend seamless continuation.
5. Ensure cumulative active-runtime accounting includes all prior attempts. Resume at lowest non-PASS milestone; re-verify its prerequisites.
6. On discrepancy, set `BLOCKED`, write reason/owner action; do not self-repair immutable evidence.
7. Use new commit/PR for documentation updates. Record **execution source commit** even if docs committed later.

## 7. Historical M1 stop report (superseded by §10, 2026-11-11)

```text
STATUS: READY_FOR_M2
NEXT_MILESTONE: M2 pending (not run; this task stopped at M1)
EXECUTION_SOURCE_BRANCH/HEAD: feat/f1rl-demo-completion / 8e7b99c4d462bdbfdf6d010e068404e03e717a08 (execution base)
CLEAN_SOURCE: true at delivery after the scoped feature-branch commit; no unrelated worktree changes
DOCS_BRANCH/HEAD: feat/f1rl-demo-completion (roadmap, evidence ledger and handoff included in the delivery commit)
ARTIFACT_ROOT: D:\Projects\.archives\F1-RL\demo-completion
MODEL_FREEZE_SHA: PENDING
TRAINED_MODEL_SHA: 09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf (smoke only)
CONFIG/PHYSICS/OBS/ACTION_SHA: frozen PPO config 8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4; Physics V4 `c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37`; Physics V5 `52e6cbca398f2132bcb96f99e77556f443bdd48338dd7c65f987b5d68c64ea68`; observation/action contracts covered by the hashed source inventory
BUDGET_EPOCH: 2026-11-10 WP3 engineering run
ACTIVE_RUNTIME_USED_SECONDS: 1365.796 provisional; prior PPO ledger 1263.031 + M1 benchmark 102.765; no PPO training/smoke/evaluation ran
ACTIVE_RUNTIME_REMAINING_SECONDS: 85034.204 provisional after charging the 102.765s M1 real benchmark; no PPO training/smoke/evaluation ran
IMPROVEMENT_CYCLES_USED: 0 by this demo campaign
LAST_GATE: M1 PASS / READY_FOR_M2. Versioned geometry/reference finish 2 passed; bounded API/WebSocket E2E and timeout regression passed; Python Physics V4/API regression 136 passed; observation/action plus Physics stability 44 passed; PPO runtime adapter 3 passed; web 105 passed; build passed. Low-load V5 benchmark: 787 ticks, 102.765s wall, 100.984s process CPU, Physics-step wall mean/p95 98.271/185.254ms, realtime factor 0.766×. No source Physics/PPO contract changed and no PPO workload ran.
BLOCKERS: None for M1 engineering gate. M2 is pending and outside this task; V5 is not admitted for training/smoke and its crash/roll terminal contract is not versioned; seven external WP7 witness sets remain absent
OPEN_PR: none
NEXT_SAFE_COMMAND: independently review `D:\Projects\.archives\F1-RL\demo-completion\m1-hardening-20261111\SHA256SUMS.json` (SHA-256 `0330c4cc74b87ef7e43a317fa6f1c003da3e5a3abc47e5b66404742f89027628`); only then begin the separately authorized M2 smoke. Do not access HELD-OUT or merge main.
OWNER_APPROVAL_NEEDED: none for this M1 remediation; stop before M2 workload, held-out evaluation, or main merge
```

## 8. Agent stop-report template (update this file at every stop)

```text
STATUS: <PASS | FAIL | BLOCKED | NOT_ACCEPTED | READY_FOR_REVIEW>
NEXT_MILESTONE: <M0..M10>
EXECUTION_SOURCE_BRANCH/HEAD: <branch> / <40-char-SHA>
CLEAN_SOURCE: <true/false>
DOCS_BRANCH/HEAD: <branch> / <40-char-SHA>
ARTIFACT_ROOT: <absolute Windows path>
MODEL_FREEZE_SHA: <hash or PENDING>
TRAINED_MODEL_SHA: <hash or PENDING>
CONFIG/PHYSICS/OBS/ACTION_SHA: <hashes or PENDING>
BUDGET_EPOCH: <id or NOT_INITIALIZED>
ACTIVE_RUNTIME_USED_SECONDS: <verified>
ACTIVE_RUNTIME_REMAINING_SECONDS: <verified>
IMPROVEMENT_CYCLES_USED: <0/1/2, independently verified>
LAST_GATE: <milestone + result + evidence SHA>
BLOCKERS: <exact source/test/artifact cause>
OPEN_PR: <URL or none>
NEXT_SAFE_COMMAND: <exact command with run limits, or NONE>
OWNER_APPROVAL_NEEDED: <specific action, or NONE>
```

## 10. M2–M3 continuation audit (2026-11-11)

```text
STATUS: M2_IN_PROGRESS / M3_BLOCKED_ON_M2 / NOT_ACCEPTED
NEXT_MILESTONE: current-source M2 real smoke with process-safe executable/PATH, then M3 fresh benchmark
EXECUTION_SOURCE_BRANCH/HEAD: feat/f1rl-demo-completion / f4b57a12428b8749db8c11d87fb00e95866df7f8
CLEAN_SOURCE: true at audit start; no active matching PPO processes
ARTIFACT_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m3-real-benchmark-20261111
MODEL_FREEZE_SHA: PENDING
TRAINED_MODEL_SHA: smoke-only 09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f
CONFIG/PHYSICS/OBS/ACTION_SHA: smoke profile 3d078a4bc45718407fb8a58cdb2f2dd856df101510deb6c9327ebff61c43ee36; Physics V4 c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37; frozen v3 config 8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4
BUDGET_EPOCH: 2026-11-10 WP3 engineering run
ACTIVE_RUNTIME_USED_SECONDS: 1371.220 verified previous-workload lower bound including 2.268s failed M3 preflight and 3.156s failed M2 launch
ACTIVE_RUNTIME_REMAINING_SECONDS: 85028.780 provisional
IMPROVEMENT_CYCLES_USED: 0
LAST_GATE: Historical 256-step smoke is internally valid but source-bound to `6452517e` and does not satisfy current-source M2 admission. Current-source attempt `smoke-20261110T230720Z-524ac611` failed before training: 0 timesteps, 0 updates, process-count watchdog; result SHA 76a5663f7596e3613e94cb983b338d6147002dda50121ed0eb4bf8b244b5493b. Dry-run diagnosis found nested Python/Git launcher processes (max tree 4 under worktree venv); direct base Python + direct Git executable stayed at max 2 under unchanged profile.
BLOCKERS: Fresh current-source M2 smoke required before M3. Dry-run process-tree summary is archived at `D:\Projects\.archives\F1-RL\demo-completion\m2-m3-current-source-20261111\process-tree-diagnostic\process-tree-diagnostic-summary.json`, SHA-256 `76e4ae850764064a95e4d43ba06cffa7622854621da3bf95c5cc34f2127ad86a`. Old M3 profile SHA 1b9f787b… is inconsistent with the benchmark profile at its declared source commit; old timing claims are historical and excluded from worker admission. V5 remains inadmissible for PPO.
OPEN_PR: none
NEXT_SAFE_COMMAND: from the clean feature branch, set `PYTHONPATH` to `D:\Projects\.worktrees\F1-RL\f1rl-demo-completion;D:\Projects\.worktrees\F1-RL\f1rl-demo-completion\.venv\Lib\site-packages`, prepend `.venv\Scripts` and `C:\Program Files\Git\ucrt64\bin` to `PATH`, then run `C:\Users\TranTien\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe scripts\wp3_real_ppo_benchmark.py smoke --output-root D:\Projects\.archives\F1-RL\demo-completion\m3-real-benchmark-20261111`. This preserves the 2-process profile and uses the shared M3 root so sequential admission can verify the smoke. Do not run 4 workers.
OWNER_APPROVAL_NEEDED: none for M2 reuse or bounded M3 rerun; stop before M6, HELD-OUT, or main merge
```

## 11. M1 final hardening closeout (2026-11-11)

```text
STATUS: M1 PASS / READY_FOR_M2 (engineering gate only)
TASK_SCOPE: Final M1 hardening; no M2/PPO work authorized or started by this closeout
EXECUTION_BASE: 8e7b99c4d462bdbfdf6d010e068404e03e717a08
CURRENT_BRANCH_HEAD_AT_CLOSEOUT: 50c76513f1225463c24bc97aa9b1d7fae4e2b65e
BRANCH: feat/f1rl-demo-completion
WORKTREE: D:\Projects\.worktrees\F1-RL\f1rl-demo-completion
M1_EVIDENCE_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m1-hardening-20261111
M1_SHA256_INVENTORY: 0330c4cc74b87ef7e43a317fa6f1c003da3e5a3abc47e5b66404742f89027628
LAST_GATE: M1 PASS. Geometry/reference finishes, bounded API/WebSocket E2E, Python and Physics V4/PPO contract regressions, web tests/build, and controlled low-load V5 benchmark are evidenced in the ledger. Fresh custody check matched all 16 source/config and 12 output hashes. No M1 test was rerun during this documentation-only closeout.
SCOPE_NOTE: A sequential M3 benchmark process was already running when this closeout began. It was stopped to honor the request to end at M1; partial attempt artifacts were preserved at D:\Projects\.archives\F1-RL\demo-completion\m3-real-benchmark-20261111\sequential-20261110T231718Z-24ff8728. Treat that attempt as incomplete; it is not M1 evidence or an accepted benchmark.
NEXT_SAFE_ACTION: None within this M1 task. Stop here.
HELD_OUT: SEALED
MAIN_MERGE: NOT PERFORMED
```

## 9. Final handoff at initial document creation (historical)

- **Next action:** M0 reconnaissance; do not start C0 full-length training yet.
- **Last historical implementation audit:** 4c8808f local parallel candidate, 63 + 3 scoped tests passed, mock campaign passed. No real benchmark or 70-run acceptance.
- **Git requirement:** documentation PR awaiting owner review, not merged; separate parallel branch apparently unpushed. Do not make Git changes to main.
- **No active training launched by this documentation authoring.** Local Windows control was disconnected during creation; local disk and current worktree state could not be freshly inspected this turn.
- **No claim of demo readiness.** Current outcome remains `NOT_ACCEPTED / NOT_EVALUATED`.

## 12. M2–M5 development continuation closeout (2026-11-11)

```text
STATUS: M2_PASS / M3_PASS_2_WORKERS / M4_ENGINEERING_PASS / M5_EVALUATION_COMPLETE_CANDIDATE_FAIL / NOT_ACCEPTED
NEXT_MILESTONE: None; stop at M5 as authorized. No M6 or final acceptance work started.
BRANCH / WORKTREE: feat/f1rl-demo-completion / D:\Projects\.worktrees\F1-RL\f1rl-demo-completion
HEAD: ba01400f09ef1487ac75f8e760d754f1583a6584
M4_M5_SOURCE_TREE_SHA256: 1cb0b80a39fed35363bdc397156aabf9191b13984d463a839663c0dce3b7d3f4
EVIDENCE_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m4-m5-development-v2
M2: current-source smoke 256 timesteps / 10 optimizer updates, 23.375s; profile 3d078a4bc45718407fb8a58cdb2f2dd856df101510deb6c9327ebff61c43ee36; independently verified.
M3: current-source sequential 2x4096 = 668.718s; two-worker 2x4096 = 340.625s; profile f3f3cad13eb6b8f717a93584b4cd5551717405a15584a26c9eb62fb34821ec5f. Four-worker benchmark not run.
M4: bounded v2 plan trained C-seed0 and A-seed0, 65,536 steps/model. Final model hashes C=600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4; A=ea2e9c69c0ec5c3bdd1a883f0ebf610127d5b10375dc791eb79063ce8f5a22f5. This is not the full 12-job v3 campaign.
M5: report SHA 34f4e923dc5ab000b6a3cd5b929e6c60afaa6d0355450d55754ef48eba407c0b; 100/100 valid episodes; no harness timeout/error; 2258s. C-seed0 mean finish 0.62 but right 0/10, ramp 1/10; A-seed0 0.00 mean finish. Candidate gate FAIL; no model freeze.
HELD_OUT: SEALED; no final acceptance tracks created; no M6+; no main merge.
NEXT_SAFE_ACTION: None within this M2–M5 continuation. Preserve all artifacts; further training requires a newly authorized continuation and a new versioned plan.
```

## 13. M5 diagnosis stop report (2026-11-11)

```text
STATUS: M5_DIAGNOSIS_COMPLETE / CANDIDATE_FAIL / REVIEW_REQUIRED / NOT_ACCEPTED
NEXT_MILESTONE: Owner review of Right Turn V3 and Cycle 1 plan; do not train before review
EXECUTION_SOURCE_BRANCH/HEAD: feat/f1rl-demo-completion / 636c4ec15b4f965e615a8055cd3f721c2b4ae942
SOURCE_TREE_SHA256: f77d5033663a3681e8b33ad96d70a993eda0d6f1f014018e84f17cb726b176af
SOURCE_CODE_UNCHANGED: true; only the three roadmap documents are updated for this handoff
ARTIFACT_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m5-diagnosis-20261111
MODEL_FREEZE_SHA: PENDING
TRAINED_MODEL_SHA: C0 600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4; A0 ea2e9c69c0ec5c3bdd1a883f0ebf610127d5b10375dc791eb79063ce8f5a22f5 (neither frozen)
CONFIG/PHYSICS/OBS/ACTION_SHA: Physics V4 c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37; action/observation sources are inventoried in SHA256-MANIFEST.json
M5_PAIRED_DIAGNOSTICS: 48/48 complete; 24 seed pairs; deterministic and stochastic; no 120s worker wall timeout; 1,083.406s grid wall
A0: deterministic full throttle+brake every step and ~0.0345m/s; stochastic adds motion but every case still fails; 174/226 curriculum-boundary terminations are a contributor hypothesis, not proven sole cause
C0_RIGHT_TURN: V1 crashes 3/3 in both modes at the road patch junction; same-seed V2 PPO comparison finishes 3/3 in both modes
C0_RAMP: deterministic 3/3 finishes in 197 steps; stochastic 0/3, step-limit 500 at 76.2–84.2% progress, no collision
RIGHT_TURN_V1_V2: V1 asset SHA a302f5d1d7f08d2b51ae963a5e695d8600d52a0ebf2d1f212ca0a4fbf90e303e; V2 SHA 5f285238821c869fd2e47e95576074b6a1ba1c7c618fa9c30c5b9f89e9103f30; details in Evidence Ledger §0.5
TESTS: geometry suite 46 PASS / 1 deselected; separately run source-lock test FAIL because env_factory.py differs from anchored Git blob; no Physics V4/PPO source contract change
V2_TRACE_HELPER: all 6 PPO environments completed; original helper failed only during EpisodeMetrics summary serialization; all six hash-verified traces were reconstructed without reruns
M4_M5_BASELINE: original M5 100/100 valid episodes; C0 0/10 right V1 and 1/10 ramp; A0 0/10 all five cells; no model freeze
IMPROVEMENT_CYCLES_USED: 0
HELD_OUT: SEALED; M2/M3 were not rerun; no training, M6+, final acceptance tracks, or main merge
EVIDENCE_REPORT_SHA256: D79A4569CB0DAC99B544AADEDA53F00763C65E2217B2CB78EB539306ACB81F7D
EVIDENCE_MANIFEST_SHA256: 73168b13cedfae17039787ecb45beb88d623b93c3ee2610d5a26acfb29349b72 (140 artifacts + 20 source/config records)
BLOCKERS: source-lock anchor drift in test_right_turn_train_v2_design; owner review needed on whether V3 supplements V1 or changes the development candidate roster
NEXT_SAFE_COMMAND: review M5-DIAGNOSIS-REPORT.md and the Cycle 1 acceptance proposal; do not start training until design/roster scope is approved
OWNER_APPROVAL_NEEDED: approve V3 design and M5 candidate-roster treatment before Cycle 1 workload; no main merge requested
```

## 14. M5 final hardening handoff (2026-11-11)

```text
STATUS: M5_ENGINEERING_FIX_PASS / M5_CANDIDATE_NOT_ACCEPTED
BRANCH: feat/f1rl-demo-completion
CODE_COMMIT: d34a58341c55540d37d8743621be5cc2b4154bec
PORTABLE_SOURCE_REVISION: V2, anchor d34a58341c55540d37d8743621be5cc2b4154bec; 44/44 source locks PASS
PORTABLE_V1: preserved; frozen historical registration unchanged
STEP3_WITNESS: restored archive source-witnesses-6e30ebd7; source commit 6e30ebd78b010d3f20d120b936c0283eab6ca78a; 4/4 hashes and provenance PASS
RIGHT_TURN_V2_PURE_PPO: 10/10 deterministic, 10/10 stochastic; diagnostics only; zero collision/timeout
RAMP_V1_PURE_PPO: deterministic 10/10 identical trajectories (not robustness); stochastic 1/10; 9/10 500-step simulation limits; zero collision/wall timeout
RAMP_STOCHASTIC_CAUSE: throttle/brake overlap >0.05 on 13.4% of steps; mean speed 2.445m/s; action SD throttle/brake/steering 0.406/0.216/0.677; evidence consistent with slow progress, not sole-cause proof
AUTO_DRIVE_RIGHT_LANE: offset capped 3.5m within 4m route corridor; focused 51 tests PASS; not PPO evidence
TESTS: source/M5 6 PASS; Auto Drive 51 PASS; Physics V4/PPO 156 PASS; source locks 44/44; STEP3 witnesses 4/4
PLAN_FILE_SHA256: db1cc1694b766fb1e92bf16892c9c5978f79cb4aa3dcfe3d2f05b3592070024e
REPORT_FILE_SHA256: c57d78397e424f6a62d2b696c5124569c508bab4333338a782d613221f6586b9
SHA256_MANIFEST: 42685f135ed1f6d74dddd4fc5bd4a0327ba3984937999d450c411a2c229f3bc6
EVIDENCE_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m5-engineering-fix-20261111\final-hardening
IMPROVEMENT_CYCLE_1: proposed only; no training. First action: non-training throttle/brake and Ramp-progress ablation; then submit a versioned single-change plan for review/authorization.
HELD_OUT: SEALED
M2_M3_RERUN / PPO_TRAINING / M6_PLUS / U_TURN: NOT RUN
MAIN_MERGE: NOT PERFORMED
NEXT: finish final diff/test/hash review; commit and push feature branch only
```

## 15. M5 Improvement Cycle 1 handoff (2026-11-11)

```text
STATUS: M5_CYCLE1_DIAG_COMPLETE / CANDIDATE_NOT_ACCEPTED / OVERALL_NOT_ACCEPTED
BRANCH: feat/f1rl-demo-completion
EVIDENCE_SOURCE_HEAD: e3b7f749186d676cbd1a63796e68d0930866cbfe
SOURCE_TREE_SHA256: 8030081db7fedd253d306cc6a31c8a365aa3d56f
RAMP_V1_STOCHASTIC: baseline 1/10; std*0.5 10/10; std*0.65 10/10
SLALOM_STOCHASTIC: baseline 10/10; std*0.5 9/10; std*0.65 6/10
OTHER_STD050_STOCHASTIC: Straight 10/10; Right Turn V2 10/10; Left Turn V2 10/10
DETERMINISTIC_DIAGNOSTICS: Ramp 10/10; Slalom 0/10; do not substitute for registered stochastic contract
CAUSE: sampled PPO action variance is a proven contributor to Ramp failures; no action mapper, Physics, reward, or step-limit defect demonstrated
CHANGE: only learned log_std scaled; no gradient update; original C0 checkpoint preserved
FINETUNING: NOT RUN; no safe training change supported after Slalom regression
VALIDATION: std*0.5 100/100; std*0.65 40/40; zero worker failures/timeouts; focused tests 12 PASS, final validator tests 2 PASS
CHECKPOINT_STD050_SHA256: b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86
CHECKPOINT_STD065_SHA256: 17d8a52f308e78c015238b9caa6f6ce4ee0e7b8289981b52b81d399e84ba6665
REPORT_STD050_SHA256: 8854e1de6b6ae5c939206389392c85185c703f82fee488d9b7ec1347ceb11616
REPORT_STD065_SHA256: 1184bd25ecef03bf161ff163e1801e12059d7435e5ceb51f45f69bc291e88212
MANIFEST_SHA256: 7ce4e93f948f7115857fb8aadf732be383e1dc6018469783224548c060c09e75 (308 artifacts)
EVIDENCE_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m5-improvement-cycle1-20261111
HELD_OUT: SEALED; M2/M3 rerun, M6+, U-turn, main merge: NOT RUN
NEXT: independent review of candidate scorecard/Slalom regression; do not claim DEMO_READY
```
