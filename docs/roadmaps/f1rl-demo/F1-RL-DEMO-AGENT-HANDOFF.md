# F1-RL — Agent Handoff / Resume Contract

> **Document version:** 2.2 (2026-11-11). **Current state:** `M0_PASS; M1_V5_FEASIBILITY_PASS; M1_FULL_ACCEPTANCE_BLOCKED; M2_PENDING`.
> This is an actionable **handoff**, not a certification of PPO driving.  
> **Authoritative goals:** [Master Roadmap](F1-RL-DEMO-COMPLETION-ROADMAP.md) · **Proof/limits:** [Evidence Ledger](F1-RL-DEMO-EVIDENCE-LEDGER.md).

## 0. Live takeover snapshot (2026-11-11)

- PR #60 and PR #61 **MERGED**. Local `main` and `origin/main` both at `ab8951a9c68f009d7b2a14127b13c07426a2460a`.
- Working branch/worktree: `feat/f1rl-demo-completion` / `D:\Projects\.worktrees\F1-RL\f1rl-demo-completion`; M1 implementation `b578d83`; audit remediation source commit `afcc26e`.
- M0 **PASS** on recorded evidence; V5 Auto Drive feasibility **PASS**; M1 full coverage **BLOCKED** because the TRAIN/development distribution lacks intersections/overpasses and combined custom geometry. The API/WebSocket V5 route with a barrier and cone finished successfully at tick 787 with zero collision IDs; this is not PPO evidence. Validation now separates observation/action, physics-profile, and Pure PPO runtime compatibility. V5 checkpoint probes establish interface compatibility only; V5 training/smoke and crash/roll terminal acceptance remain closed. P2-03 isolated-route timing/profile found physics stepping dominant, but the reported full-suite timing variance remains unexplained. No final demo acceptance or new PPO campaign. External WP7 evidence and scientific Gate 2 remain separately blocked/sealed.
- Previous real smoke/benchmark source: `8582d9d08826074dbd2b6d32609d9d58639f771d`; five model SHA-256 files matched. New merged benchmark metadata fix changes the source hash, without changing past artifacts.
- Active runtime from all seven historical attempts plus one independently run post-merge smoke: **1,263.031 / 86,400 seconds**, remaining **85,136.969 seconds** (provisional pending any additional earlier workloads). Two improvement cycles remain unused by this demo campaign.
- On `main` `ab8951a`, independent scoped regression: **121 passed, 2 skipped**. Five historical checkpoints loaded, hashes matched and predicted finite actions. Post-merge new real smoke from `6452517` PASS 256 steps / 10 optimizer updates / checkpoint `09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf`; 90.718 seconds and preserved under `D:\Projects\.archives\F1-RL\wp3-postmerge-real-smoke-20261110`.
- **Perf observation:** new smoke rollout 85.255 seconds vs historical 18.040 seconds; cause unknown, avoid speedup/ETA claim and investigate in M0/M3 as needed.
- Any historical claims below about unmerged docs/parallel branches are **superseded by this dated snapshot**. No scientific HELD-OUT access.

## 1. Current operator-readable status

| Key | Status / fact |
|---|---|
| Overall | `NOT_ACCEPTED` — M1 full coverage and real Pure PPO acceptance are incomplete |
| Next milestone | **Complete M1 TRAIN/development geometry coverage**; M2 remains pending |
| M0-M10 | M0 PASS; M1 V5 feasibility PASS; M1 full coverage BLOCKED; M2-M10 PENDING |
| Operating permission | Autonomous within owner-approved scope and 24h active workload cap; never merge main |
| Scientific HELD-OUT | **SEALED**; WP7/Gate 2 **BLOCKED** |
| Final model freeze | Not created |
| Three custom acceptance tracks | Not created — must only be generated **after** model freeze |
| 70-run Pure PPO evaluation | Not performed |
| Live real-time 1×/30FPS/p95≤150ms | Not measured |
| Training budget epoch | 2026-11-10 WP3 engineering run + 1 new post-merge smoke: **1,263.031 s** used; **85,136.969 s** remaining (provisional) |
| Agent improvement cycles | 0 newly used **by this documentation task**; confirm pre-existing active campaign history before charging |
| Remaining gates | M1 intersection/overpass TRAIN/development coverage; reconcile broad-suite performance variance; then M2 admission/smoke and later milestones; V5 remains custom Auto Drive only; seven external WP7 engineering nodes still lack witnesses |

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

## 7. Current stop report (2026-11-11)

```text
STATUS: BLOCKED
NEXT_MILESTONE: M1 full geometry coverage
EXECUTION_SOURCE_BRANCH/HEAD: feat/f1rl-demo-completion / afcc26e (audit remediation source commit)
CLEAN_SOURCE: clean after source commit afcc26e and the documentation commit containing this handoff
DOCS_BRANCH/HEAD: feat/f1rl-demo-completion (this handoff is in the latest documentation commit)
ARTIFACT_ROOT: D:\Projects\.archives\F1-RL\demo-completion
MODEL_FREEZE_SHA: PENDING
TRAINED_MODEL_SHA: 09de1596d324f27bb39716c5032c88c4071f23f0b18c4343390da8beebb5f5bf (smoke only)
CONFIG/PHYSICS/OBS/ACTION_SHA: config 8a67db80464dc3b93072c0edc431f7a71b1b271c392b47d0915af0f2cac106e4; Physics V4 `fe6e...`; observation/action manifest recorded in smoke artifact
BUDGET_EPOCH: 2026-11-10 WP3 engineering run
ACTIVE_RUNTIME_USED_SECONDS: 1263.031 prior PPO workload ledger; this turn ran no PPO training/smoke/benchmark/evaluation
ACTIVE_RUNTIME_REMAINING_SECONDS: 85136.969 prior PPO ledger, provisional; M1 engineering Auto Drive probe is not charged as PPO workload
IMPROVEMENT_CYCLES_USED: 0 by this demo campaign
LAST_GATE: V5 Auto Drive feasibility PASS; M1 full coverage BLOCKED. P1-02 compatibility regression passed; layered API/WebSocket route with cone/barrier passed to finish and had zero collision IDs; 52 focused Python tests and 105 web tests passed; build passed. One-core route timing: 787 ticks, 107.726s wall, 101.391s process CPU; mean environment step 135.431ms, p95 236.396ms. Physics integration dominates; broad-suite timing variance remains unexplained. No PPO training or smoke run.
BLOCKERS: TRAIN/development intersection/overpass and combined custom geometry coverage is absent; broad-suite performance variance is not attributed; V5 is not admitted for training/smoke and its crash/roll terminal contract is not versioned; seven external WP7 witness sets remain absent
OPEN_PR: none
NEXT_SAFE_COMMAND: add a versioned TRAIN/development intersection/overpass geometry fixture and validate a combined-feature route; do not start M2 until M1 full acceptance is met. If pursuing P2-03, rerun the exact broad suite in an otherwise idle host and compare CPU time against the archived one-core route profile.
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

## 9. Final handoff at initial document creation (historical)

- **Next action:** M0 reconnaissance; do not start C0 full-length training yet.
- **Last historical implementation audit:** 4c8808f local parallel candidate, 63 + 3 scoped tests passed, mock campaign passed. No real benchmark or 70-run acceptance.
- **Git requirement:** documentation PR awaiting owner review, not merged; separate parallel branch apparently unpushed. Do not make Git changes to main.
- **No active training launched by this documentation authoring.** Local Windows control was disconnected during creation; local disk and current worktree state could not be freshly inspected this turn.
- **No claim of demo readiness.** Current outcome remains `NOT_ACCEPTED / NOT_EVALUATED`.
