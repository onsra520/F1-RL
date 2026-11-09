# Physics SAT Correctness and Worker Benchmarks Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the Physics V4 SAT broad-phase false negative and add bounded pytest and environment-worker benchmarks without opening scientific admission.

**Architecture:** Keep exact OBB SAT authoritative. Match its normalized separating direction and raw support-basis arithmetic; keep raw body basis in the conservative broad-phase AABB. Use pytest-xdist explicitly for test processes and keep the five timeout-sensitive nodes in one registered resource group, run serially at acceptance. Compare existing SB3 vector backends directly on a synthetic legacy F1RLEnv without importing `TrainingRunner` or a PPO model.

**Tech Stack:** Python 3.11+, pytest, pytest-xdist, Gymnasium, Stable-Baselines3 vector environments, PowerShell process metrics on the Windows host.

**Spec:** `docs/superpowers/specs/2026-11-09-physics-sat-workers-design.md`

## Global Constraints

- Keep the `1e-9 m` SAT threshold and all Physics, collision, reward, observation, and action contracts unchanged.
- Do not retrain PPO, call the training runner, create checkpoints, run HELD-OUT/scientific evaluation, or bypass admission.
- Do not update frozen source locks, change `main`, or merge.
- Run full regression only after SAT differential checks, worker smoke, and environment-only benchmark pass.
- Keep Auto Drive/T05 `session.wait(120)` unchanged and run the five sensitive nodes serially for full acceptance.
- Capture structured logs with a trace ID per independent benchmark run; store diagnostic artifacts outside the repository.

## Review Focus

- A rotated body axis whose norm rounds above 1 near the `1e-9 m` boundary must not be rejected before exact SAT — Task 1 regression test.
- Non-finite or unusable query axes must take the existing conservative/fail-closed path — Task 1 differential invalid-input test.
- The five timeout-sensitive nodes must remain unchanged in assertions and timeout while grouped for worker scheduling — Task 2 exact-node smoke.
- `SubprocVecEnv(spawn)` workers must close on success and failure without leaking child processes or sharing output artifacts — Tasks 2 and 3 worker cleanup checks.
- No benchmark may call `TrainingRunner`, allocate a PPO model, write checkpoints, or imply research admission — Task 3 static checks and output contract.

---

### Task 1: SAT boundary correctness and P1/P2 performance comparison

**Files:**
- Modify: `f1rl/sandbox/swept_surface_clearance.py`
- Modify: `tests/sandbox/test_bounded_patch_clearance.py`
- Read-only baseline: `main` at `07d54c7fc3ba8502f13b568350be4c13bcb61be2`
- Benchmark reference: `docs/reports/PHYSICS-V4-INDEPENDENT-PERFORMANCE-REAUDIT-20261109.md`

**Interfaces:**
- Preserve `_BoundedPatchIndex.has_separation_at_most(body_box, clearance_threshold_m) -> bool`.
- Preserve `obb_separation_lower_bound()` as the final SAT verdict.

- [x] Add `test_indexed_sat_matches_exhaustive_at_1e9m_rounded_body_axis` using the one-patch values in the design doc. Assert exhaustive gap `<= 1e-9`, candidate inclusion, and indexed verdict `True`.
- [x] **RED:** Run `D:\Projects\F1-RL\.venv\Scripts\python.exe -m pytest tests/sandbox/test_bounded_patch_clearance.py::test_indexed_sat_matches_exhaustive_at_1e9m_rounded_body_axis -q` from the feature worktree. Expected: FAIL because the candidate is present but indexed verdict is `False`.
- [x] Normalize each finite, nonzero body axis once before the query and fast gap projection, matching exhaustive SAT normalization. Preserve all-candidate fallback/fail-closed behavior for unsupported values; do not alter the threshold.
- [x] **GREEN:** Run `D:\Projects\F1-RL\.venv\Scripts\python.exe -m pytest tests/sandbox/test_bounded_patch_clearance.py tests/sandbox/test_body_collision.py -q`. Expected: all focused SAT tests PASS.
- [x] Add deterministic differential coverage for rotated, slope, seam, overhang, grazing, near-threshold, and invalid-query cases. Expected: zero candidate misses, indexed/exhaustive verdict differences, or unhandled exceptions.
- [x] Run the external seeded differential harness at `D:\Projects\.archives\F1-RL-sat-workers-20261109\sat_diff_diagnostic.py` with 100,000 cases. Expected: zero candidate misses and zero verdict mismatches; save structured output with a trace ID.
- [x] Use the exact node `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]`, fixture `tests/runtime/fixtures/t05-original-obstacle-track.json` (SHA-256 `3359bae959535baa40d337ae7677540ab64f022e739acd148fa2f9b65369ec37`), seed 412, 477 steps. The report's P0/P1/P2 timings are historical context. Run three alternating repetitions on `07d54c7` and three on the corrected branch. Keep the 120-second wait. Expected: all six runs finish; record source hashes, wall/CPU time, and steps/s; investigate repeatable material slowdown before proceeding.

### Task 2: pytest-xdist and resource-sensitive test scheduling

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements-test.txt`
- Modify: `tests/runtime/test_auto_drive_integration.py`
- Modify: `tests/runtime/test_t05_hitbox_regression.py`
- Read-only worker contracts: `tests/learning/training/test_vector_env.py`, `tests/learning/training/test_pv5_backend_admission.py`, `tests/learning/training/test_training_runner.py`

**Interfaces:**
- Keep worker count explicit on the pytest CLI (`-n 1`, `-n 2`, `-n 4`); do not default to `auto`.
- Use registered `resource_sensitive` and shared `xdist_group` markers on only the five exact nodes listed in the design doc. The default pytest configuration uses `--dist=loadgroup`; worker count remains an explicit CLI choice.

- [x] Add pytest-xdist to development/test dependencies and register `resource_sensitive`; do not change runtime dependencies.
- [x] Mark the two parameterized obstacle nodes, T05 centerline node, and both T05 hitbox nodes. Assertions and all existing waits remain unchanged.
- [x] **Smoke, 1 worker:** 59 focused SAT/vector/runtime cases PASS in 8.97s.
- [x] **Smoke, 2 workers:** same 59 cases PASS in 8.05s.
- [x] **Smoke, 4 workers:** same 59 cases PASS in 6.63s.
- [x] Collect-only confirms exactly the five resource-sensitive node IDs. Full resource group PASS on baseline in 199.47s and candidate in 190.22s; all 120-second waits unchanged.
- [x] Worker cleanup checked after each xdist smoke: no pytest/execnet processes remained. Smoke scope uses no shared writable outputs; runtime managers close in `finally`.

### Task 3: PPO vector-environment benchmark only

**Files:**
- Modify if needed: `scripts/benchmark_vector_envs.py`
- Add focused CLI contract test if output changes: `tests/learning/training/test_vector_env_benchmark.py`
- External process metrics and logs: `D:\Projects\.archives\F1-RL-sat-workers-20261109\`

**Interfaces:**
- Compare `DummyVecEnv` with `SubprocVecEnv(start_method="spawn")` around a synthetic legacy `F1RLEnv`; avoid the training package and runner imports entirely.
- Report `training_permitted=false`, `heldout_permitted=false`, `learning_called=false`, and `checkpoint_created=false`; do not construct a PPO model.

- [x] Added the CLI contract test first; it failed on missing admission/checksum fields, then passed after the bounded diagnostic update.
- [x] Ran exactly 1,024 transitions for dummy and subproc at 1/2/4 envs, three repetitions each (18/18 PASS).
- [x] External psutil process-tree instrumentation recorded wall throughput, CPU, aggregate RSS, checksums, and child cleanup. Matching backend/count checksums were identical; all permission/learning/checkpoint fields were false; no child processes leaked.
- [x] Static inspection confirmed the harness does not import `TrainingRunner` or PPO model APIs and directly uses synthetic legacy envs only.

### Task 4: Baseline comparison, full regression, and report

**Files:**
- Create: `docs/reports/PHYSICS-V4-SAT-WORKERS-BENCHMARK-20261109.md`
- Read-only baseline: `main` at `07d54c7fc3ba8502f13b568350be4c13bcb61be2`

- [x] Collected complete scope: 3,332 test nodes on baseline and 3,337 on candidate.
- [x] Static policy inventory identified 101 nodes to mark `BLOCKED_BY_POLICY`: tests that call `TrainingRunner`, any `.learn()` method, successful/checkpoint `.save()`, or saved-policy evaluation. No test was changed; exact IDs are in the report appendix.
- [x] The 5 resource-sensitive nodes PASS serially on baseline and candidate.
- [x] Run the remaining 3,226 baseline and 3,231 candidate nodes with four workers, deselecting only the 101 policy-blocked nodes and the already-run five resource-sensitive nodes. Capture exact statuses, wall time, process-tree CPU/RSS, test IDs, and hashes.
- [x] Compare outcomes exactly. Report `NOT_READY` and never claim full regression pass while 101 nodes remain `BLOCKED_BY_POLICY`.

No commit, push, merge, PPO learning, checkpoint creation, or branch switch is part of this plan.
