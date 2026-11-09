# Physics V4 SAT Correctness and Worker Benchmark Report

| Field | Value |
|---|---|
| Date | 2026-11-09 |
| Verdict | `NOT_READY` for full-regression acceptance; implementation and evidence are ready for review |
| Repository | `D:\Projects\F1-RL` |
| Baseline | `main` at `07d54c7fc3ba8502f13b568350be4c13bcb61be2` |
| Candidate | `codex/sat-broadphase-workers`, separate worktree based on the same HEAD |
| Environment | Windows, Python 3.12.15, pytest 9.2.0, pytest-xdist 3.8.0 |

## 1. Scope and repository state

The SAT fix, pytest-xdist configuration, and bounded vector-environment benchmark were implemented only in the candidate worktree. `main` stayed clean at the baseline commit. No branch switch, commit, push, merge, PPO learning, checkpoint creation, HELD-OUT access, scientific evaluation, or source-binding update occurred.

The candidate worktree remains uncommitted. Source/test changes are listed in §8. Diagnostic scripts, logs, JUnit XML, process metrics, and differential output are under `D:\Projects\.archives\F1-RL-sat-workers-20261109`.

**Source hashes:**

- Baseline `f1rl/sandbox/swept_surface_clearance.py`: `1e1440a6ad9778b9435cf3c3e4e21c567f97a5eb0edfc996bb3be4b104a440fb`
- Candidate same file: `095761b5e3c132f3ad2e8a05f63ec841b0d507018566df109fa85889c8b09405`
- The WP7 registered physics/source locks were not changed. This Physics V4 source hash change does require source-bound owner review before any scientific acceptance.

## 2. SAT correctness

The minimized `1e-9 m` repro demonstrated a real false-negative in the fast SAT prefilter. The exhaustive kernel normalized each separating direction, while the fast prefilter used a quaternion-rotated body axis without normalization. At the boundary, the exhaustive gap was `9.999999781306063e-10 m` but the fast filter computed `1.0000000128250756e-9 m` and rejected the contact.

The candidate normalizes only the SAT query directions. It preserves the raw rotated body basis for support radii and broad-phase AABB construction, matching the exhaustive kernel’s floating-point operation order. The `1e-9 m` threshold is unchanged; exhaustive SAT remains the final verdict; invalid query geometry retains the conservative/fail-closed path. An intermediate version that normalized the support basis was rejected after differential testing found 349 verdict mismatches and five near-threshold false negatives.

**Verification:**

- `python -m pytest tests/sandbox/test_bounded_patch_clearance.py tests/sandbox/test_body_collision.py -q` — **31 PASS**, 2.87 s.
- External differential, seed `20261110`: 100,000 rotated/slope queries plus 31,488 near-threshold queries at `1e-9 m`; 15,630 exhaustive eligible contacts; **0** candidate-inclusion misses, **0** indexed/exhaustive verdict mismatches, **0** false negatives, **0** exceptions. Trace `084589a9c11846a9bd81dec9d7b72ab`.
- The three added SAT regression cases passed in the candidate full regression. The two new `tests/learning/training/test_vector_env_benchmark.py` CLI contract tests also passed.

This establishes correctness for the tested numerical/adversarial set, not a proof over every possible floating-point input. Keep the exhaustive differential and fail-closed path as acceptance requirements.

## 3. Auto Drive/T05 timeout group

These five exact nodes were marked `resource_sensitive` and run serially, preserving each existing `session.wait(120)` contract:

1. `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]`
2. `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]`
3. `tests/runtime/test_auto_drive_integration.py::test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve`
4. `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]`
5. `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]`

**Result:** baseline **5/5 PASS**; candidate **5/5 PASS**. Baseline group: 199.469 s wall / 195.938 s CPU. Candidate: 190.222 s wall / 186.984 s CPU. This is one group run per revision and should not be interpreted as a stable speedup.

The paired T05 normal-45.0 episode was measured three times per revision, sequentially, with the same 477-step fixture (SHA-256 `3359bae959535baa40d337ae7677540ab64f022e739acd148fa2f9b65369ec37`), seed 412, and unchanged 120-second limit:

| Revision | Wall s | CPU s | Steps/s |
|---|---:|---:|---:|
| Baseline run 1 | 63.077 | 61.734 | 7.562 |
| Baseline run 2 | 65.624 | 63.500 | 7.269 |
| Baseline run 3 | 64.786 | 63.109 | 7.363 |
| Candidate run 1 | 61.551 | 60.125 | 7.750 |
| Candidate run 2 | 64.733 | 62.906 | 7.369 |
| Candidate run 3 | 64.138 | 62.516 | 7.437 |

Median baseline was 64.786 s / 63.109 CPU s / 7.363 steps/s; candidate was 64.138 s / 62.516 CPU s / 7.437 steps/s. The ~1% difference is within run-to-run variation. The SAT correction is a correctness fix; this benchmark does not establish a meaningful Physics speed optimization.

Historical P0→P1/P2 measurements from `PHYSICS-V4-INDEPENDENT-PERFORMANCE-REAUDIT-20261109.md` remain context only: P0 73.623 s wall / 70.703 s CPU; retained P1+P2 runs 61.585/58.828 and 61.891/59.016. Those earlier optimizations were already present at this baseline and were not reimplemented here.

## 4. Pytest worker smoke and runtime

`pytest-xdist` is constrained to `<4` and added to development/test dependencies. Worker count is explicit; pytest uses `--dist=loadgroup`. The five timeout-sensitive tests remain a serial resource group.

The same 59 focused SAT/vector/runtime cases passed at each setting:

| Mode | Result | Wall |
|---|---:|---:|
| `-n 1` | 59 PASS | 8.97 s |
| `-n 2` | 59 PASS | 8.05 s |
| `-n 4` | 59 PASS | 6.63 s |
| serial (`-n 0`) | 59 PASS | 8.23 s |

Four workers were 1.24× the serial throughput on this small selection; one worker incurred process-start overhead. Worker cleanup was checked after the smokes and after the full runs; no pytest/execnet workers remained. The bounded tests use isolated temporary outputs and did not share writable test artifacts.

### Full regression, with policy boundary

The complete collection was 3,332 baseline and 3,337 candidate nodes. A fail-closed external collection filter excluded 101 nodes per revision that call `TrainingRunner`, `.learn()`, checkpoint-producing `.save()`, or saved-policy evaluation. No tests were edited to force pass. The five timeout-sensitive nodes were already run serially and were not included in the worker remainder.

| Revision | 4-worker remainder | PASS | FAIL | SKIP | BLOCKED_BY_POLICY | Serial resource PASS | Integrated total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3,226 | 3,130 | 93 | 3 | 101 | 5 | 3,332 |
| Candidate | 3,231 | 3,133 | 95 | 3 | 101 | 5 | 3,337 |

Exact worker-run command shape: `python -m pytest --rootdir=. -n 4 --dist=loadgroup -q -p audit_policy_filter --audit-policy-manifest <external-manifest> --audit-revision <baseline|candidate> --junitxml=<external-output>`. The temporary plugin verified every expected policy node and the five serial-resource nodes before deselecting anything.

Structured operation trace IDs: full baseline `d30738c6-7825-456d-a758-5d02ceb82ee8`; full candidate `990e9991-2955-436f-8ddb-83af6de12037`; serial resource groups `ab6d7fa26f6a4dfbb22bf1a753a65249` and `bc8b858b0eb7496d90ec928455804856`; 18-run vector benchmark suite `9812cc438319460c8af906d9b79f8f82`.

- Baseline: 735.006 s wall, approximately 2,730.5 s sampled process-tree CPU, 2,769 MiB peak process-tree RSS.
- Candidate: 996.159 s wall, approximately 3,834.0 s sampled process-tree CPU, 2,860 MiB peak process-tree RSS.
- Candidate full-run wall time was 35.5% above baseline (996.159 s vs 735.006 s). The user later confirmed a game was consuming substantial CPU/GPU from 22:43–23:01 on 2026-11-09, overlapping the candidate run. Treat that comparison as contaminated: it cannot establish a code regression or a reliable speedup. The earlier fixture-scheduling explanation was an unverified hypothesis. See the stable same-revision scheduler re-benchmark in §10.
- A follow-up acceptance run should compare `--dist=loadscope` or explicit groups for expensive module fixtures; keep the five timeout-sensitive nodes separate and serial. No scheduler change is made in this report.

The 93 baseline failures recur exactly on candidate. Their causes are 77 missing external `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` artifacts (**ENVIRONMENT**), 14 pre-existing frozen WP7 source-hash mismatches (**CONTRACT_MISMATCH**), and two unrelated `tests/test_run.py` failures. Both `test_run.py` nodes were rerun serially and failed on both baseline and candidate, so they are not xdist races. Candidate has two additional WP2 run-binding test failures because those tests require a clean checkout while the authorized candidate diff is uncommitted; the fail-closed clean-source contract worked and no binding was produced. Full failure IDs and summaries are in the details appendix.

All three SKIPs are Windows symlink tests skipped because symlink creation is unavailable (`WinError 1314`). No full-regression PASS is claimed: 101 nodes remain `BLOCKED_BY_POLICY`, and baseline/environment failures remain unresolved.

## 5. PPO vector backend benchmark (benchmark-only)

The harness uses a synthetic legacy `F1RLEnv` and compares direct `DummyVecEnv` and `SubprocVecEnv(start_method="spawn")` stepping. It does not import or call `TrainingRunner`, PPO model APIs, `.learn()`, a training factory, or checkpoint writes. Each of the 18 runs executed exactly 1,024 transitions; output fields kept training/held-out permissions false. Checksums matched across backends at each environment count, and no child process leaked.

Median of three repetitions per cell:

| Envs | Backend | End-to-end transitions/s | Step-loop transitions/s | Wall s | Process-tree CPU s | Peak RSS MiB |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Dummy | 5,915.9 | 5,935.1 | 1.984 | 1.500 | 213.24 |
| 1 | Subproc spawn | 475.4 | 2,633.7 | 4.071 | 3.047 | 420.88 |
| 2 | Dummy | 5,969.9 | 6,000.7 | 2.064 | 1.484 | 212.90 |
| 2 | Subproc spawn | 476.6 | 4,860.5 | 4.084 | 4.625 | 628.61 |
| 4 | Dummy | 6,210.3 | 6,251.3 | 2.001 | 1.578 | 213.36 |
| 4 | Subproc spawn | 417.6 | 7,439.9 | 4.466 | 7.969 | 1,044.18 |

At four environments, subprocess step-loop throughput is ~1.19× higher, but end-to-end throughput is ~15× lower, CPU ~5× higher, and RSS ~4.9× higher because spawn/import overhead dominates this small bounded workload. This does not justify adding runtime workers or changing PPO defaults. No PPO training or retraining is recommended or performed.

## 6. Source-binding and scientific-integrity risks

The SAT fix changes the hash of `swept_surface_clearance.py`; any aggregate Physics V4 protocol hash that includes this file changes too. Scientific acceptance must therefore review the new source hash and physics semantics before any future scientific run. The source lock, research guards, reward, observation, action contract, and admission code remain untouched; admission stays closed.

The full suite exposes 14 pre-existing WP7 source-binding mismatches on baseline. Thirteen identify `f1rl/research/wp7_track_feasibility.py`: Git blob SHA-256 `a0b6b53080f6e48d07993bbafb33da0e88537f40efb3afc0ea2ca8e10f5e64ab`, registered Windows SHA `18ffc3050fe278cd62a9992ae2a62aef8876d577f4aabbfb10d9b9f5f4f55ab0`, current checkout SHA-256 `a72c33d0a215fedf41be4a730f8bcb746d65b3a5ea3ca6b3ab29ab904b6609ba`. One additional Stage-2 source-binding test finds `f1rl/learning/algorithms/ppo.py` at canonical-LF SHA `68ff35f0c144bf2f8f4e75d1d06ff7f2da2ea2d35e9e65c71f2e22b0aa242d2e` while the proposal expects `d17b6edd30068673d5cf156cbf601a63a7e43e43726ad118981b05aafe84c4e6`. These failures already occur on clean baseline `main`; this work changed neither file nor any lock.

## 7. Minimal follow-up and acceptance order

1. Physics owner reviews the SAT arithmetic change and differential cases. Preserve the unchanged `1e-9 m` threshold, exhaustive final SAT, and conservative invalid-input behavior.
2. Benchmark the full remainder using module-aware worker scheduling (`--dist=loadscope` or explicit groups) to avoid duplicate module-scoped fixture work. Keep the five Auto Drive/T05 nodes serial.
3. Provision the external WP3 evidence for tests that require it, or report those nodes as environment-blocked; do not fabricate or rewrite evidence.
4. Resolve the existing WP7 pinned-source mismatch through its owner-controlled source-binding process. Do not rewrite frozen locks as part of this audit.
5. Review the two baseline `tests/test_run.py` failures separately. Revisit the two WP2 clean-tree failures only in an approved clean review checkout; preserve the test’s fail-closed source-cleanliness rule.
6. Keep the 101 learning/checkpoint/evaluation nodes blocked until a separate policy decision. Never report `FULL_REGRESSION_PASS` while they remain blocked.

**Acceptance criteria for this candidate:** zero SAT differential mismatches; focused and 59-case worker smoke PASS at 1/2/4 workers; all five timeout nodes PASS serially under the existing timeout; no xdist process leaks; no changes to admission/source locks. These criteria are met. **Overall `NOT_READY`** is retained because full-regression status still includes 101 policy blocks and 95 candidate FAIL nodes, many inherited from baseline/environment.

## 8. Changed files

- `f1rl/sandbox/swept_surface_clearance.py`
- `pyproject.toml`
- `requirements-test.txt`
- `scripts/benchmark_vector_envs.py`
- `tests/runtime/test_auto_drive_integration.py`
- `tests/runtime/test_t05_hitbox_regression.py`
- `tests/sandbox/test_bounded_patch_clearance.py`
- `tests/learning/training/test_vector_env_benchmark.py`
- `docs/superpowers/specs/2026-11-09-physics-sat-workers-design.md`
- `docs/superpowers/plans/2026-11-09-physics-sat-workers.md`
- `docs/reports/PHYSICS-V4-SAT-WORKERS-BENCHMARK-20261109.md`
- `docs/reports/PHYSICS-V4-SAT-WORKERS-DETAILS-20261109.md`

`git diff --check` passed. Main remains clean; no commit or push was made.

## 9. Evidence locations

- Full collection logs, policy-node manifest, JUnit XML, stdout/stderr, timings, differential diagnostics, resource-group runs, worker smoke, and 18 vector benchmark runs: `D:\Projects\.archives\F1-RL-sat-workers-20261109`.
- Exact failed node IDs, skip reasons, and all 101 policy-blocked node IDs: [`PHYSICS-V4-SAT-WORKERS-DETAILS-20261109.md`](PHYSICS-V4-SAT-WORKERS-DETAILS-20261109.md).
- Design decisions: [`2026-11-09-physics-sat-workers-design.md`](../superpowers/specs/2026-11-09-physics-sat-workers-design.md).

## 10. Stable worker scheduler re-benchmark (2026-11-09/10)

**Benchmark verdict: `READY_FOR_REVIEW` for the scheduler comparison.** The overall full-regression verdict above remains `NOT_READY`; this addendum does not resolve or rerun the 95 candidate regression failures or the 101 policy-blocked nodes.

### Setup and load checks

All ten runs used candidate worktree `codex/sat-broadphase-workers` at exact Git HEAD `07d54c7fc3ba8502f13b568350be4c13bcb61be2`, with unchanged Physics source SHA-256 `095761b5e3c132f3ad2e8a05f63ec841b0d507018566df109fa85889c8b09405`. Only the scheduler flags changed. Every run used the same 37 collected tests:

- `tests/sandbox/test_bounded_patch_clearance.py`
- `tests/sandbox/test_body_collision.py`
- `tests/learning/training/test_vector_env_benchmark.py`
- `tests/research/test_wp7_spatial_profile_execution.py::test_every_actual_train_has_same_registered_spatial_physics_finish_and_stop` (four parameter cases)

This selection does not run PPO learning or create checkpoints. The expensive module-scoped `nominal` fixture performs deterministic TRAIN spatial execution checks and dominates elapsed time. The exact target set was run twice serially and twice for each xdist setting, in balanced order. xdist defaults were neutralized with `-o addopts=`; worker runs used explicit `--dist=loadscope` or `--dist=loadgroup`. `-n 0` ran without xdist.

The 23-second initial preflight measured CPU at 8.3–11.3% (9.8% average), RAM at 39.9–40.7%, and GPU at 1%; no game process was present in the sampled process list. Before each of the ten runs, the guard also passed: average CPU ranged 4.72–12.30%, highest CPU sample 16.3%, RAM 39.8–43.6%, minimum available memory 26.87 GiB, and maximum sampled GPU utilization 8%. Guard limits were CPU average 20%, any CPU sample 35%, RAM 82%, and GPU 15%. A separate Playwright/Chromium installation process appeared briefly during the final serial run; three system CPU samples remained 2.40–7.05%, memory commitment 45.16–45.25%, and that process had exited by post-run inspection. No high-load or game overlap was observed in this re-benchmark.

### Results

Wall and CPU values are seconds; RSS is peak sampled process-tree memory. Fixture values show the median total setup time across worker instances and the number of setup instances per run. Speedup is calculated against the two-run serial median (104.887 s). Every configuration passed 37/37 tests in both runs, with zero failures, errors, or skips; all worker cleanup checks passed.

| Configuration | Wall median (range) | Speedup vs serial | CPU median | Peak RSS median | `nominal` setup instances/run | Fixture setup total median |
|---|---:|---:|---:|---:|---:|---:|
| `-n 0` serial | 104.887 (103.409–106.364) | 1.000× | 98.586 | 464.0 MiB | 1 | 95.697 s |
| `-n 2 --dist=loadscope` | **102.403 (100.756–104.050)** | **1.024×** | 103.758 | 934.7 MiB | 1 | 98.178 s |
| `-n 4 --dist=loadscope` | 106.219 (104.889–107.548) | 0.987× | 110.547 | 1,402.7 MiB | 1 | 102.167 s |
| `-n 2 --dist=loadgroup` | 111.983 (106.835–117.131) | 0.937× | 208.102 | 968.1 MiB | 2 | 205.527 s |
| `-n 4 --dist=loadgroup` | 108.592 (103.991–113.192) | 0.967× | 263.750 | 1,639.2 MiB | 2–3 | 258.756 s |

The measured median winner was `-n 2 --dist=loadscope`, 2.4% faster than serial. Its run range overlaps the serial range (103.409–104.050 s), and only two repetitions were taken, so this is not strong evidence of a dependable speedup. `loadscope` consistently ran the module fixture once. `loadgroup` duplicated it in both repeats, approximately doubling fixture setup and process-tree CPU; four workers duplicated it two or three times. Four-worker `loadscope` was slightly slower than serial and used about 3× the peak RSS. For this representative suite, use serial when minimizing resource use; if workers are required, `-n 2 --dist=loadscope` is the measured choice. Avoid `loadgroup` for this module-fixture-heavy selection.

### Per-run outcomes and evidence

| Configuration | Run 1 wall / CPU / RSS | Run 2 wall / CPU / RSS | Fixture instances (run 1 / run 2) | Result |
|---|---:|---:|---:|---|
| Serial | 106.364 / 99.828 / 468.1 MiB | 103.409 / 97.344 / 459.8 MiB | 1 / 1 | 37/37 PASS; cleanup PASS |
| 2, `loadscope` | 100.756 / 101.984 / 928.8 MiB | 104.050 / 105.531 / 940.5 MiB | 1 / 1 | 37/37 PASS; cleanup PASS |
| 4, `loadscope` | 107.548 / 112.422 / 1,403.7 MiB | 104.889 / 108.672 / 1,401.6 MiB | 1 / 1 | 37/37 PASS; cleanup PASS |
| 2, `loadgroup` | 117.131 / 218.000 / 967.7 MiB | 106.835 / 198.203 / 968.4 MiB | 2 / 2 | 37/37 PASS; cleanup PASS |
| 4, `loadgroup` | 113.192 / 322.047 / 1,657.0 MiB | 103.991 / 205.453 / 1,621.3 MiB | 3 / 2 | 37/37 PASS; cleanup PASS |

The runner wrote a separate metrics JSON, JUnit XML, stdout/stderr, preflight sample, and trace ID for all ten runs under `D:\Projects\.archives\F1-RL-sat-workers-20261109`. Its final summary-writing step exited with `TypeError: log() got multiple values for argument 'event'` after the tenth pytest run. This was a diagnostic harness reporting error; each recorded pytest process returned 0, all ten metrics JSON files are complete, and the per-run evidence above was aggregated directly from those files. No source, test, dependency, configuration, or scientific-binding file was changed by this re-benchmark; this report is the only file updated for the present task.
