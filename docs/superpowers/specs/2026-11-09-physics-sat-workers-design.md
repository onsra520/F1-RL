# Physics SAT correctness and worker benchmark design

**Date:** 2026-11-09
**Branch/base:** `codex/sat-broadphase-workers` from `main` at `07d54c7fc3ba8502f13b568350be4c13bcb61be2`

## Intent and boundaries

Fix the demonstrated Physics V4 SAT broad-phase false negative while keeping the registered `1e-9 m` contact threshold, exact SAT kernel, rewards, observations, actions, and fail-closed behavior. Add bounded pytest parallelism and compare `DummyVecEnv` with the existing `SubprocVecEnv(spawn)` using environment-only benchmarks.

No PPO learning/retraining, training-runner calls, checkpoint creation, HELD-OUT, scientific evaluation, admission changes, frozen source-lock updates, merges, or changes on `main`.

## Current evidence

In `swept_surface_clearance.py`, the fast SAT prefilter originally compared a raw quaternion-rotated body axis as though it were unit length, while `obb_separation_lower_bound()` normalizes the SAT axis. A one-patch repro at the unchanged `1e-9 m` threshold has exhaustive gap `9.999999781306063e-10 m`, raw fast-filter gap `1.0000000128250756e-9 m`, candidate `[0]`, and indexed verdict `False`.

A first correction normalized both the separating direction and the body support basis. Differential testing then exposed a second rounding mismatch: an exhaustive gap of `9.909058490276653 m` was rejected by a fast gap of `9.909058490276655 m` (two ULP). The exhaustive kernel normalizes the separating direction but calculates support using its original raw rotated basis vectors. The accepted correction now mirrors that operation order: use normalized directions for SAT comparisons, raw rotated basis vectors for support radii and spatial AABB construction, then call exhaustive SAT for the final verdict.

The optimization reference is `docs/reports/PHYSICS-V4-INDEPENDENT-PERFORMANCE-REAUDIT-20261109.md`, §10. Its single-run P0 baseline was 73.623 s wall / 70.703 s CPU; retained P1+P2 T05-normal runs were 61.585 s / 58.828 s and 61.891 s / 59.016 s. The exact benchmark node is `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]`, using `tests/runtime/fixtures/t05-original-obstacle-track.json` (SHA-256 `3359bae959535baa40d337ae7677540ab64f022e739acd148fa2f9b65369ec37`), seed 412, 477 steps. Historical phase timings are context, not a controlled paired comparison.

The repository already supports `SubprocVecEnv(start_method="spawn")`; `configs/training/vectorized-ppo.yml` selects eight `DummyVecEnv` slots. `TrainingRunner` rejects Physics V4 scientific-admission training. That guard stays untouched. The test dependency set does not include pytest-xdist.

## Approaches and selected choices

| Area | Options | Selected |
|---|---|---|
| SAT filter | Normalize all body basis vectors for support; add a numeric cushion; or reproduce the exhaustive kernel's axis/support arithmetic. | Normalize only SAT separating directions. Use the same raw rotated support basis and AABB basis as the exhaustive kernel. Keep the unchanged threshold and exact SAT as final verdict. |
| Pytest parallelism | pytest-xdist; or a repository-specific process scheduler. A custom scheduler adds lifecycle and artifact-isolation code. | Add pytest-xdist as a test/development dependency; select worker count explicitly. Mark the five Auto Drive/T05 timeout nodes as one resource group and run that group serially for full acceptance. The documented `loadgroup` scheduler keeps same-marked tests on one worker ([pytest-xdist docs](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)). |
| PPO worker benchmark | Build a new executor; measure the configured training factory; or directly compare the existing SB3 vector backends around a synthetic environment. | Directly benchmark `DummyVecEnv` and `SubprocVecEnv(spawn)` around a bounded synthetic legacy `F1RLEnv`. This avoids importing `TrainingRunner`, PPO models, training factories, or admission pathways. Report per-step checksum and closed admission fields. |

## Planned changes

- `f1rl/sandbox/swept_surface_clearance.py`: normalize SAT query directions, retain raw rotated body basis vectors for support radii and broad-phase AABBs, and keep exhaustive SAT as the verdict; do not widen `clearance_threshold_m`.
- `tests/sandbox/test_bounded_patch_clearance.py`: add the minimal `1e-9 m` regression and seeded indexed-versus-exhaustive differential cases, including near-threshold, rotated, seam, grazing, slope, overhang, and invalid-query cases.
- `pyproject.toml` and `requirements-test.txt`: add pytest-xdist and register markers/configuration needed for explicit worker runs.
- `tests/runtime/test_auto_drive_integration.py` and `tests/runtime/test_t05_hitbox_regression.py`: mark only the five known timeout-sensitive nodes as a shared xdist group; preserve all existing timeout values and assertions.
- `scripts/benchmark_vector_envs.py` and `tests/learning/training/test_vector_env_benchmark.py`: compare environment-only throughput for dummy and spawn workers. The CLI runs exactly the bounded transition count, returns a deterministic SHA-256 checksum, records `training_permitted=false`, `heldout_permitted=false`, `learning_called=false`, and `checkpoint_created=false`; an external process-tree sampler records wall time, CPU/RSS, and worker cleanup.
- `docs/reports/`: record exact commands, baseline comparison, per-test PASS/FAIL/SKIP, performance, source hashes, and remaining admission/provenance limits.

The physics source change changes a source-bound file hash. Frozen locks remain unchanged, so this work does not authorize scientific use or retraining.

## Verification and acceptance

1. On the baseline source, the new regression reproduces the false negative; on the fix, indexed and exhaustive verdicts agree. The final external differential measured 100,000 rotated/sloped queries plus 31,488 near-threshold cases at `1e-9 m`: zero candidate-inclusion misses, zero indexed/exhaustive mismatches, zero exceptions, and zero threshold false negatives. An intermediate all-normalized-support attempt had 349 mismatches and five boundary false negatives and was rejected before the final fix.
2. Use the audit report above for the recorded before/after P0→P1→P2 measurements. For a controlled correctness-fix comparison, run the exact T05 `normal-45.0` node three times on `07d54c7` and three times on the fixed branch, in alternating order; keep `session.wait(120)` and report wall time, CPU time, steps/s, and fixture/source hashes. Investigate any repeatable material slowdown before proceeding.
3. Run representative worker smoke at 1, 2, and 4 pytest workers. Check repeatability, isolation, timeouts, shared artifact paths, race failures, and worker shutdown. Keep the five timeout-sensitive nodes serial in full acceptance.
4. Eighteen environment-only runs (three repetitions per backend at 1/2/4 envs) had stable matching checksums, closed admission flags, and no worker leaks. At 4 envs Subproc step-loop throughput improved 19% in this tiny synthetic fixture, while end-to-end throughput was 15x lower, process-tree CPU about 5x higher, and RSS about 4.9x higher due to spawn/import overhead. No PPO worker/default changes are authorized by this benchmark.
5. The full regression collected 3,332 baseline / 3,337 candidate tests. A static policy audit identified 101 nodes that directly/indirectly invoke `TrainingRunner`, `.learn()`, checkpoint-producing `.save()`, or saved-policy evaluation; these are `BLOCKED_BY_POLICY` and remain unchanged. Run the 5 timeout-sensitive nodes serially (5/5 PASS on both revisions), then all remaining nodes at four workers. Compare exact node IDs and classify PASS, FAIL, SKIP, BLOCKED_BY_POLICY, and pre-existing baseline failures. Never call this `FULL_REGRESSION_PASS` while policy-blocked nodes remain. The 12–20 minute goal is informational.

**Builder-review decision:** pin the T05 normal node and fixture above; the earlier report already contains the P0/P1/P2 phase values, while the new paired run isolates the SAT correction cost.

**Ready criteria:** no SAT false negatives in the stated differential scope; worker smoke and cleanup pass; no new full-regression failures versus baseline; no guard, lock, threshold, or training changes. Otherwise verdict is `NOT_READY` with exact blockers.
