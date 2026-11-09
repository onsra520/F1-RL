# F1-RL targeted audit implementation report

**Date:** 2026-11-09  
**Repository:** `D:\Projects\F1-RL`  
**Branch / HEAD:** `main` / `6f0110a871087828a9c9ce16327281bafbabf4d6`

## 1. Git state and scope

HEAD and branch are unchanged. The worktree has five modified tracked files and two untracked reports: the pre-existing `TARGETED-AUDIT-REPORT.md` and this report. No commit, push, merge, or branch switch was made. `git diff --check` passed. Temporary audit logs and the pytest instrumentation are outside the repository under `D:\Projects\.archives\F1-RL-targeted-audit-20261109`.

No full regression, PPO training/retraining, HELD-OUT run, registered scientific evaluation, evidence rewrite, or source-binding edit was performed. The five DEVELOPMENT fixtures were run serially with their existing 120-second waits.

## 2. Seven original failures and classification

| Original failed test ID | Classification | Final result |
|---|---|---|
| `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]` | `PERFORMANCE` | PASS |
| `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]` | `PERFORMANCE` | PASS |
| `tests/runtime/test_auto_drive_integration.py::test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve` | `PERFORMANCE` | PASS |
| `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]` | `PERFORMANCE` | PASS |
| `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]` | `PERFORMANCE` | PASS |
| `tests/learning/training/test_env_factory.py::test_research_admission_marker_reaches_created_environment` | `TEST_OBSOLETE` | Removed stale expectation; canonical factory tests pass |
| `tests/learning/training/test_env_factory.py::test_factory_rejects_wrong_constructor_field_types[research_admission_required-yes]` | `TEST_OBSOLETE` | Removed stale parameter; canonical factory tests pass |

The five runtime failures were throughput timeouts, not controller stalls, deadlocks, or bad terminal/contact outcomes. Baseline sessions consumed about 97–98% of one CPU core, advanced continuously, and reached only 0.18–0.64 route progress before `wait(120)` expired. `cancelled` was recorded during timeout cleanup. The final sessions reached `finished`, completed the route, and reported zero collisions.

The factory tests expected `research_admission_required`, a field absent from Băng’s canonical handoff (`ce0e55d7a428826b2d4ed3218641105bd0206f4b`). That field and both test expectations were introduced together in `6cdead0`; the later consolidation restored the handoff source but retained those tests. Canonical `F1RLEnvFactory` and `F1RLEnv` derive `requires_scientific_admission` from explicit body geometry or non-legacy physics configuration. No production caller sets the extra marker. Only the two stale test expectations were removed; `env_factory.py`, `gym_environment.py`, and their research guards were left unchanged.

## 3. Reproduction and benchmark

The exact five tests were run one at a time with the existing 120-second waits. Each invocation used:

```text
uv run --extra dev python D:\Projects\.archives\F1-RL-targeted-audit-20261109\run_test.py <exact pytest node ID> <JSONL trace path>
```

The runner recorded a trace ID at session start and structured progress/end events, including wall/CPU time, steps, position, route fraction, speed, collision count, terminal result, and timings for physics, controller, navigation, collision sweeps, and replay serialization.

| Fixture | Before: wall / CPU / steps/s | After: wall / CPU / steps/s | Final steps / terminal |
|---|---:|---:|---|
| Cone | 120.352 s / 116.906 s / 1.770 | 18.256 s / 18.000 s / 16.926 | 309 / finish |
| Barrier | 120.458 s / 117.703 s / 1.818 | 17.758 s / 17.391 s / 17.400 | 309 / finish |
| T05 centerline | 120.058 s / 117.031 s / 0.850 | 56.306 s / 54.609 s / 8.187 | 461 / finish |
| T05 normal | 121.104 s / 118.094 s / 0.735 | 84.728 s / 82.844 s / 5.630 | 477 / finish |
| T05 racing | 121.325 s / 118.391 s / 0.725 | 84.434 s / 82.125 s / 5.377 | 454 / finish |

All five final test invocations passed under 120 seconds. Relative to the measured baseline, throughput improved about 7.4–9.6×. The T05 normal and racing tests also passed their original minimum hitbox-separation assertion (`> 0.2 m`); the centerline test passed its lateral-bound assertion. All five replays ended with route progress 1.0 and collision count 0.

Terminal state was `finished` / `finish` for every fixture. Custom DEVELOPMENT fixtures had no checkpoints. Final positions and speeds were cone `(3.000, 0.500, 200.611) m / 9.374 m/s`, barrier `(3.000, 0.500, 200.889) m / 9.374 m/s`, T05 centerline `(99.410, 0.500, -34.059) m / 9.052 m/s`, T05 normal `(101.730, 0.501, -34.075) m / 7.633 m/s`, and T05 racing `(101.760, 0.495, -33.735) m / 8.471 m/s`.

One-step `cProfile` benchmark (wall time includes profiler overhead):

| Profile fixture | Before | After | Change |
|---|---:|---:|---:|
| TRAIN Slalom, 1 surface | 0.124 s | 0.115 s | approximately unchanged |
| DEVELOPMENT speed-cycle, 101 surfaces | 1.746 s | 0.132 s | 13.2× faster |
| DEVELOPMENT T05, 142 surfaces | 3.014 s | 0.169 s | 17.8× faster |

The final per-session traces show the remaining cost is Physics v4 integration. For example, T05 normal spent 66.1 s in the v4 integrator, versus 0.52 s in controller prediction, 1.33 s in route projection, 0.010 s in frame serialization, and 0.114 s in replay serialization. T05 racing spent 66.7 s in v4 integration. The Auto Drive cone session spent 12.4 s in v4 integration, 0.22 s in controller prediction, 0.59 s in route projection, and 0.15 s in replay serialization. This identifies the timed-out work as CPU-bound physics, not Auto Drive, navigation, replay, or thread lifecycle.

Obstacle sweep checks were also timed: cone 4,991 calls / 1.14 s; barrier 4,991 / 1.11 s; T05 centerline 11,280 / 2.58 s; T05 normal 15,896 / 3.72 s; T05 racing 15,422 / 3.62 s. No collision was recorded. Full road-patch SAT kernel calls were 5,503 for T05 normal and 6,276 for T05 racing after the body-axis rejection; the remaining certificate work is included in the v4 integrator totals above.

The final trace IDs are `fd1278929c604d57b55e3d5f1a6bf1f1` (cone), `d801507ab63848df8a425a710ee7acfd` (barrier), `363857e940834228827d47ca496bf5a3` (T05 centerline), `73b3e19ac63449889037e3a6b6510aa6` (T05 normal), and `3c2da637964544239bef01014f5cdf37` (T05 racing). Their JSONL records and the one-step profile are in the external audit archive named above.

## 4. Minimal implementation

The Physics v4 integration builds one bounded-patch index for the authored atlas and reuses it for the chassis sweep certificate and finite tire-footprint queries. The spatial query uses a conservative grid. A fast body-face projection check rejects a patch only when one of the exact SAT body axes proves its gap already exceeds the requested threshold; remaining candidates still use the existing 15-axis SAT. Tire coverage keeps the same polygon kernel, ordered patch selection, height calculation, bump/droop checks, and forces; the index only limits which polygons with possible positive coverage reach that kernel. Invalid or unbounded queries fall back to all patches, and unresolved sweeps remain fail-closed.

Added differential tests compare indexed SAT threshold results with exhaustive SAT over rotated poses and thresholds, compare indexed tire coverage with exhaustive coverage, preserve clear/contact sweep verdicts, and assert fallback behavior for non-finite queries. The index is enabled only for `atlas_rotating_edge_v4`; the v3 regression module passed.

Files changed:

- `f1rl/sandbox/swept_surface_clearance.py`
- `f1rl/sandbox/finite_tire_footprint_v4.py`
- `f1rl/sandbox/spatial_contact_dynamics.py`
- `tests/sandbox/test_bounded_patch_clearance.py`
- `tests/learning/training/test_env_factory.py` (obsolete marker-only expectations removed)
- `TARGETED-AUDIT-IMPLEMENTATION-REPORT.md` (this report)

No reward, PPO, controller, timeout, or scientific-guard source changed.

## 5. Focused verification

The final focused module command was run with the project’s uv development environment. The small bootstrap installs the local `tests` directory as a namespace package because the environment also has a third-party package named `tests`:

```text
uv run --extra dev python -c "import sys,types,pytest; package=types.ModuleType('tests'); package.__path__=['D:/Projects/F1-RL/tests']; sys.modules['tests']=package; raise SystemExit(pytest.main(['-q','tests/sandbox/test_bounded_patch_clearance.py','tests/sandbox/test_spatial_contact_dynamics.py','tests/sandbox/test_spatial_atlas_lifecycle.py','tests/sandbox/test_spatial_compliant_lifecycle.py','tests/sandbox/test_spatial_rotating_lifecycle.py','tests/sandbox/test_unilateral_spatial_contact.py','tests/learning/training/test_env_factory.py']))"
```

Result: **76 passed in 16.71 s**. This includes the factory contract, v4 bounded clearance, atlas/contact lifecycle, v3 rotating lifecycle, and unilateral contact modules.

The exact TRAIN Slalom baseline was run as `tests/runtime/test_auto_drive_integration.py::test_right_hand_auto_drive_finishes_slalom_without_collision`: **PASS in 16.00 s**, 318 steps, all 3 checkpoints completed, route progress 1.0, collision count 0. Its trace ID is `70df18f50b434f4db53e7f03daed0751`.

No full regression suite was run.

## 6. Source-binding and provenance risk

No frozen registry, historical evidence, source hash record, or research guard was edited. Current source hashes are:

| File | HEAD source SHA-256 | Worktree SHA-256 |
|---|---|---|
| `spatial_contact_dynamics.py` | `cc9842f91b09ee79ab03509a5adbcd299637b0ac7a27e9ee09c03fb851cf8c40` | `99b7b16d915d1919dc6f2a0aa9eedfaba79d2d6facc1e8353e4243b064172213` |
| `swept_surface_clearance.py` | `a538fa2749bf1604adec0d8ae73bf6af3719e43d91c3e4e563e01c74fed688ad` | `9ff1d11fa07a21cb591f4367654aa456748665db32ed575074a3361f21a2765a` |
| `env_factory.py` | `d439369b6a7c8153f91eb3f9e207b7e36080c0590a8783fcc0cd28b2e8bc48f1` | unchanged |
| `gym_environment.py` | `9bf2be0531ea23297e84ccf2a63c97b4338e288dbca45251283ab7c3ff6563db` | unchanged |

The frozen WP3 technical registry pins `spatial_contact_dynamics.py` to `62895c5ddf042135dc9e10e8d143f3d696cbf7b0d84d8fa0abeb7d2b97a3d32c`, which already differed from this exact HEAD before the present implementation. The WP3 source-bound acceptance file also includes both modified physics files; the modified finite tire footprint implementation is not listed in that fixed source bundle. Therefore the existing scientific/source-bound evidence cannot sign off these current bytes. This is a provenance blocker before future scientific use: the physics owner must review the implementation and approve the appropriate source-binding/acceptance update. No scientific evaluation was run to bypass it.

The factory/gym source hashes remain unchanged. Removing the stale tests therefore does not alter source bindings or the derived, fail-closed admission contract.

## 7. PPO and worker conclusion

PPO retraining is **not required** for this performance-only optimization. It changes candidate lookup work while preserving the exact geometric verdict path and tire-coverage kernel; it does not change physics parameters, reward, observations, or action mapping. The change still needs the source-bound physics review above before scientific claims rely on it.

No runtime or training workers were added. A physics step mutates one sequential simulation state and consumes one CPU core; extra workers cannot shorten that single-session dependency chain. Parallel episode workers would affect aggregate throughput and add scheduling/CPU contention, but the five required tests now pass serially within their existing timeout, so worker architecture is not justified by this audit.

## 8. Priorities and acceptance criteria

1. **Physics/source-bound review:** review the conservative bounds and current source hashes before any scientific use. Accept when adversarial rotated/sloped/near-edge geometry cases still match exhaustive SAT and exhaustive tire coverage; all invalid/oversized-query paths remain fail-closed; and the owner approves the registered source bundle, including the finite tire footprint dependency.
2. **Keep the performance regression gates:** all five exact test IDs above pass with `wait(120)` unchanged; T05 tests retain their positive hitbox margin and route assertions; TRAIN Slalom completes all three checkpoints with zero collisions.
3. **Keep canonical factory contract:** no `research_admission_required` marker is introduced unless Băng’s canonical source contract is explicitly revised. Admission remains derived from typed physics/body configuration and its existing guards.
4. **No PPO retrain:** retain frozen models/evidence; do not claim a new scientific result from this engineering benchmark.

**Overall:** all seven original failures are addressed for the targeted engineering suite: the five performance timeouts now pass, and the two obsolete contract expectations are removed. The remaining blocker is source-bound physics review/registration before scientific use.
