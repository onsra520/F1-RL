# F1-RL — Agent Handoff / Resume Contract

> **Document version:** 2.5 (2026-11-11). **Current state:** `M0_PASS; M1_PASS; M2_IN_PROGRESS; M3_BLOCKED_ON_M2; OVERALL_NOT_ACCEPTED`.
> This is an actionable **handoff**, not a certification of PPO driving.  
> **Authoritative goals:** [Master Roadmap](F1-RL-DEMO-COMPLETION-ROADMAP.md) · **Proof/limits:** [Evidence Ledger](F1-RL-DEMO-EVIDENCE-LEDGER.md).

## 0. Live takeover snapshot (2026-11-11)

- PR #60 and PR #61 **MERGED**. Local `main` and `origin/main` both at `ab8951a9c68f009d7b2a14127b13c07426a2460a`.
- Working branch/worktree: `feat/f1rl-demo-completion` / `D:\Projects\.worktrees\F1-RL\f1rl-demo-completion`; current execution source at audit start `f4b57a12428b8749db8c11d87fb00e95866df7f8`.
- M0 **PASS** on recorded evidence; M1 engineering **PASS / READY_FOR_M2** after versioned development geometry, reference Auto Drive, bounded API/WebSocket finish, Physics V4/PPO regressions, web tests/build and a low-load V5 benchmark. M2 **PASS / REUSED_COMPATIBLE** from the source-bound 256-step C-seed0 smoke. M3 is **IN_PROGRESS** because old benchmark attempts bind an unreconciled profile SHA. V5 remains custom Auto Drive only; no V5 PPO, crash/roll terminal acceptance, held-out access or main merge.
- Historical real smoke source `6452517e3e887ee83f9d98b3323e3e5089d4c187`; artifact hashes and telemetry independently match. Historical M3 source `8582d9d08826074dbd2b6d32609d9d58639f771d`; its profile SHA `1b9f…` does not match the benchmark profile at that commit, so old timing results do not admit worker count.
- Active runtime lower bound including prior campaigns and M1 benchmark: **1,365.796 / 86,400 seconds**, provisional remaining **85,034.204 seconds**. M2 evidence reuse adds no time. Two improvement cycles remain unused.
- On `main` `ab8951a`, independent scoped regression: **121 passed, 2 skipped**. Five historical checkpoints loaded, hashes matched and predicted finite actions. Post-merge new real smoke from `6452517` PASS 256 steps / 10 optimizer updates / checkpoint `09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf`; 90.718 seconds and preserved under `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110`.
- **Perf observation:** new smoke rollout 85.255 seconds vs historical 18.040 seconds; cause unknown, avoid speedup/ETA claim and investigate in M0/M3 as needed.
- Any historical claims below about unmerged docs/parallel branches are **superseded by this dated snapshot**. No scientific HELD-OUT access.

## 1. Current operator-readable status

| Key | Status / fact |
|---|---|
| Overall | `NOT_ACCEPTED` — M1 and M2 are ready; M3, full PPO training and final demo acceptance remain incomplete |
| Next milestone | **M2 current-source real PPO smoke**; historical source cannot satisfy current-source admission |
| M0-M10 | M0 PASS; M1 PASS; M2 IN_PROGRESS; M3 BLOCKED_ON_M2; M4-M10 PENDING |
| Operating permission | Autonomous within owner-approved scope and 24h active workload cap; never merge main |
| Scientific HELD-OUT | **SEALED**; WP7/Gate 2 **BLOCKED** |
| Final model freeze | Not created |
| Three custom acceptance tracks | Not created — must only be generated **after** model freeze |
| 70-run Pure PPO evaluation | Not performed |
| Live real-time 1×/30FPS/p95≤150ms | Not measured |
| Training budget epoch | 2026-11-10 WP3 engineering run + M1 V5 benchmark + M3 preflight: **1,368.064 s** used; **85,031.936 s** remaining (provisional) |
| Agent improvement cycles | 0 newly used **by this documentation task**; confirm pre-existing active campaign history before charging |
| Remaining gates | Fresh M3 benchmark, M4 training, M5 development evaluation and later milestones; V5 remains custom Auto Drive only; seven external WP7 engineering nodes still lack witnesses |

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
NEXT_MILESTONE: current-source M2 real smoke, then M3 fresh benchmark
EXECUTION_SOURCE_BRANCH/HEAD: feat/f1rl-demo-completion / f4b57a12428b8749db8c11d87fb00e95866df7f8
CLEAN_SOURCE: true at audit start; no active matching PPO processes
ARTIFACT_ROOT: D:\Projects\.archives\F1-RL\demo-completion\m3-real-benchmark-20261111
MODEL_FREEZE_SHA: PENDING
TRAINED_MODEL_SHA: smoke-only 09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f
CONFIG/PHYSICS/OBS/ACTION_SHA: smoke profile 3d078a4bc45718407fb8a58cdb2f2dd856df101510deb6c9327ebff61c43ee36; Physics V4 c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37; frozen v3 config 8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4
BUDGET_EPOCH: 2026-11-10 WP3 engineering run
ACTIVE_RUNTIME_USED_SECONDS: 1368.064 verified previous-workload lower bound including 2.268s failed M3 preflight
ACTIVE_RUNTIME_REMAINING_SECONDS: 85031.936 provisional
IMPROVEMENT_CYCLES_USED: 0
LAST_GATE: Prior 256-step smoke artifact is internally valid: 10 optimizer updates, finite changed weights, checkpoint SHA 09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f and telemetry SHA 55c09d6ff8a87a064ff5b8c69db996836eb1853a934851fddba129199d0a2755. It is bound to historical source `6452517e` and does not satisfy current-source M2 admission. First M3 preflight correctly rejected without starting a child; raw preflight SHA 50ef77509963ee872aa5df7eae30e1dd7705f5493b6da2bfd9c90935e5e91284.
BLOCKERS: M2 current-source smoke is required before M3. Old M3 profile SHA 1b9f787b… is inconsistent with the benchmark profile at its declared source commit; old timing claims are historical and excluded from worker admission. V5 remains inadmissible for PPO.
OPEN_PR: none
NEXT_SAFE_COMMAND: run `.venv\Scripts\python.exe scripts\wp3_real_ppo_benchmark.py smoke --output-root D:\Projects\.archives\F1-RL\demo-completion\m2-m3-current-source-20261111` from the clean feature branch at `90bfe06`. Preserve the failed M3 preflight and do not run 4 workers.
OWNER_APPROVAL_NEEDED: none for M2 reuse or bounded M3 rerun; stop before M6, HELD-OUT, or main merge
```

## 9. Final handoff at initial document creation (historical)

- **Next action:** M0 reconnaissance; do not start C0 full-length training yet.
- **Last historical implementation audit:** 4c8808f local parallel candidate, 63 + 3 scoped tests passed, mock campaign passed. No real benchmark or 70-run acceptance.
- **Git requirement:** documentation PR awaiting owner review, not merged; separate parallel branch apparently unpushed. Do not make Git changes to main.
- **No active training launched by this documentation authoring.** Local Windows control was disconnected during creation; local disk and current worktree state could not be freshly inspected this turn.
- **No claim of demo readiness.** Current outcome remains `NOT_ACCEPTED / NOT_EVALUATED`.
