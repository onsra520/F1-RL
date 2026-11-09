# Physics v4 Independent Performance Re-Audit

**Date:** 2026-11-09 | **Repository:** `D:\Projects\F1-RL` | **Branch / HEAD:** `main` / `6f0110a871087828a9c9ce16327281bafbabf4d6` | **Current verdict:** `READY_FOR_REVIEW` (initial audit verdict `NEED_MORE_EVIDENCE`; superseded in §10)

## 1. Scope and Git state

This audit reran the five exact Auto Drive/T05 node IDs serially with their existing `session.wait(120)`, profiled the T05 normal fixture at early, middle, and late steps, compared the bounded spatial queries with exhaustive geometry, reviewed the factory contract against Băng’s handoff, and ran the focused modules and TRAIN Slalom baseline. It did not run HELD-OUT, scientific evaluation, PPO training, or full regression.

At audit start, `main` was at the HEAD shown above. The pre-existing worktree had these changes:

| Path | Pre-existing diff |
|---|---:|
| `f1rl/sandbox/finite_tire_footprint_v4.py` | 7 insertions, 1 deletion |
| `f1rl/sandbox/spatial_contact_dynamics.py` | 7 insertions, 3 deletions |
| `f1rl/sandbox/swept_surface_clearance.py` | 220 insertions, 17 deletions |
| `tests/learning/training/test_env_factory.py` | 9 deletions |
| `tests/sandbox/test_bounded_patch_clearance.py` | 122 insertions, 2 deletions |
| `TARGETED-AUDIT-IMPLEMENTATION-REPORT.md`, `TARGETED-AUDIT-REPORT.md` | pre-existing untracked reports |

The audit added this report only. `git diff --check` passed. No source, test, guard, evidence, or source-binding file was changed by this audit. Temporary scripts and logs are in `D:\Projects\.archives\F1-RL-targeted-audit-20261109`.

## 2. Seven failures and independent test results

The five runtime IDs were run individually and sequentially using:

```text
.\.venv\Scripts\python.exe -m pytest -q --tb=short <one exact node ID>
```

Each test retains its own 120-second session wait. The first uninstrumented pass and independent repeats were:

| Exact test ID | Classification | First run | Repeat | Outcome |
|---|---|---:|---:|---|
| `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]` | `PERFORMANCE` | 17.917 s | 21.332 s / 19.109 s child CPU | PASS / PASS |
| `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]` | `PERFORMANCE` | 18.435 s | 19.547 s / 17.875 s child CPU | PASS / PASS |
| `tests/runtime/test_auto_drive_integration.py::test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve` | `PERFORMANCE` | 55.299 s | 54.536 s / 51.891 s child CPU | PASS / PASS |
| `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]` | `PERFORMANCE` | 75.543 s / 72.98 s child CPU | 73.623 s / 70.703 s child CPU | PASS / PASS |
| `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]` | `PERFORMANCE` | 74.525 s / 71.75 s child CPU | Not repeated uninstrumented | PASS |

The first five current-worktree runs all passed under the unchanged timeout. T05 normal completed 477 steps in 75.543 s (6.31 steps/s); racing completed 454 steps in 74.525 s (6.09 steps/s); centerline completed 461 steps in 55.299 s (8.33 steps/s). The normal repeat was 6.48 steps/s. The normal profile ended at route fraction 1.0, position `(101.7300, 0.5009, -34.0745)`, speed `7.6333 m/s`, status `finished`, with the runtime thread stopped. TRAIN Slalom also completed and passed, detailed in §7.

The first-run wrapper measured its launcher rather than the Python child for the first three CPU totals; those CPU values are deliberately omitted. Repeat sidecar records measured the child process. T05 normal and racing first-run child CPU values came from process samples.

The two removed factory expectations classify as `TEST_OBSOLETE` (their expected marker is a `CONTRACT_MISMATCH` with the canonical implementation):

| Exact former test ID | Classification | Current contract evidence |
|---|---|---|
| `tests/learning/training/test_env_factory.py::test_research_admission_marker_reaches_created_environment` | `TEST_OBSOLETE` | Current canonical API has no `research_admission_required` field. |
| `tests/learning/training/test_env_factory.py::test_factory_rejects_wrong_constructor_field_types[research_admission_required-yes]` | `TEST_OBSOLETE` | The removed field has no constructor validation contract to test. |

The canonical factory and environment derive `requires_scientific_admission` from typed physics configuration or explicit body geometry and keep registration/admission checks fail-closed. The Băng handoff at commit `ce0e55d7a428826b2d4ed3218641105bd0206f4b` describes registering the new physics/profile/version and wiring the factory/protocol; it does not require the marker. The marker and the two expectations were introduced together in `6cdead0`. The current factory module passed as part of the focused run: 9 tests passed. No admission guard was weakened.

## 3. Episode profile and performance cause

The T05 normal exact fixture was profiled at physics steps 1, 61, 120, 180, 240, 300, 360, 420, and 460. These are per-step cProfile wall/CPU measurements, so they include profiler overhead and should be used to locate work, not as uninstrumented step latency.

| Step | Route fraction / speed at nearby progress sample | Profiled integration wall / CPU | `advance()` | RK stages | Energy refinements |
|---:|---|---:|---:|---:|---:|
| 1 | start | 0.121 / 0.109 s | 16 | 81 | — |
| 61 | 0.11 / 8.14 m/s | 0.284 / 0.281 s | 40 | 201 | 12 |
| 120 | 0.23 / 5.99 m/s | 0.662 / 0.641 s | 112 | 561 | 48 |
| 180 | between 0.23 and 0.49 | 0.270 / 0.270 s | 36 | 181 | — |
| 240 | 0.49 / about 3.8 m/s | 0.454 / 0.453 s | 64 | 321 | 24 |
| 300 | between 0.49 and 0.70 | 0.160 / 0.170 s | 16 | 81 | — |
| 360 | 0.70 / about 8.2 m/s | 0.362 / 0.344 s | 60 | 301 | 22 |
| 420 | between 0.70 and 0.95 | 0.160 / 0.160 s | 16 | 81 | — |
| 460 | 0.95 / about 8.7 m/s | 0.609 / 0.594 s | 82 | 411 | 33 |

The dominant **within-episode** cost variation is adaptive energy-ledger refinement in `spatial_contact_dynamics.py` (`advance()`, around lines 734–766). When the energy residual exceeds its configured tolerance, the integrator recursively splits the interval. At the sampled steps, 12–48 such splits expanded the base 16 advances to 40–112 advances and expanded the four-stage RK work accordingly. At steps 61, 120, 240, 360, and 460, the peak residual was respectively about 142×, 3,199×, 2,323×, 1,423×, and 3,839× the registered tolerance; the maximum observed recursion depth was 5–6. No suspension-resolution, hard-stop, edge-overtravel, roll-envelope, phase-split, or collision exception was recorded. This is state-dependent workload, not a monotonic increase with route progress. Loosening the energy tolerance or capping this refinement would change the Physics v4 integration contract and is not a safe performance fix.

The static road data is rebuilt at every call to `_integrate_spatial()` (`spatial_contact_dynamics.py:184–206`): `_atlas()` is called once and `_build_bounded_patch_index()` is called once for every physics step. For the 142-patch T05 track, each atlas build performs 10,011 pair intersections. The surface patches, seam tolerance, and body geometry remain fixed during the episode.

No-profiler setup measurements on this track gave `_atlas()` median 16.525 ms (24 repetitions), index construction median 3.070 ms (24 repetitions), and combined setup median 19.379 ms (12 repetitions). Avoiding 476 repeated builds in a 477-step session has an estimated saving of about 9.3 s. That is roughly 12% of the current 74–75 s episode, useful but too small to explain the full historical 120 s versus 85 s gap by itself.

The current broad-phase removes most patch candidates in the measured T05 states:

| Physics step | Road queries / candidates | Road candidates vs exhaustive | Tire queries / candidates | Tire candidates vs exhaustive |
|---:|---:|---:|---:|---:|
| 1 | 64 / 384 | 4.2% retained | 324 / 972 | 2.1% retained |
| 120 | 256 / 3,328 | 9.2% retained | 2,244 / 10,098 | 3.2% retained |

The query index therefore filters about 91–96% of road patch candidates and 97–98% of tire patch candidates in these samples. A second conservative body-axis projection can reject candidates before the more expensive OBB SAT lower-bound kernel. The indexed path still sends retained candidates through the existing exact checks.

The integration profile also includes RK stages, finite tire coverage/polygon clipping, and swept chassis certification. At step 120, cProfile attributed about 0.413 s cumulative to `stage()`, 0.174 s to finite tire coverage, 0.151 s to swept chassis certification, and 0.044 s to `_atlas()`; these nested cumulative values overlap and must not be added. At step 460, finite tire coverage/polygon clipping was the largest profiled secondary cost. Controller, navigation, frame capture, and replay serialization were small in the archived completed T05 trace: controller prediction 0.521 s, route projection 1.334 s, frame serialization 0.010 s, replay capture 0.304 s, and final replay serialization 0.114 s over the whole run. The physics integrator accounted for 66.130 s there.

The user-provided step-61 numbers (`_integrate_spatial` about 1.463 s and `_atlas` about 0.258 s) were not reproduced. This audit recorded the same 40 advances, 201 stages, and 10,011 intersections at step 61, with cProfile integration 0.284 s and `_atlas` 0.042 s; direct no-profiler atlas timing was about 0.0165 s. The full cProfile run took 78.607 s wall / 76.25 s CPU, about 5.4% slower than the two-run uninstrumented T05 normal mean. The selected-step branch tracing runs finished in 72–75 s. The earlier profile’s instrumentation and machine conditions are not present in its records, so its larger timing cannot be attributed precisely.

## 4. Why the archived run passed in about 85 s and another timed out

The archive contains both outcomes for the same T05 normal test ID:

| Trace | Wall / CPU | Steps and terminal | Physics v4 integration | `obb_separation_lower_bound` calls |
|---|---:|---|---:|---:|
| `final-t05-normal.jsonl` | 120.151 / 116.969 s | 443; route 0.904; cancelled after wait timeout | 103.085 s | 762,900 |
| `final2-t05-normal.jsonl` | 84.728 / 82.844 s | 477; route 1.0; finished | 66.130 s | 5,503 |

The archived pytest instrumentation maps `physics.v4.patch_sat_kernel` directly to `swept_surface_clearance.obb_separation_lower_bound`. Its count fell about 138× between these two traces, while the surrounding `has_separation_at_most` query count stayed similar (58,368 versus 63,584). The failed trace spent 36.955 s more in the integrator despite stopping 34 steps earlier. Controller, navigation, collision sweep, and replay serialization totals stayed around fractions of a second to a few seconds and do not explain the delta. This is evidence of a much more expensive geometry execution path in the failed trace.

The archive does not record source hashes or enough machine/clock telemetry per run to prove that these two executions used identical source bytes and environment. The large kernel-call difference is consistent with different source/path instrumentation or optimization state; it is not evidence that the same measured path randomly varied by 138×. The earlier 3/5 failure report also did not identify its three IDs in a trace available here. This audit reran all five exact IDs on the current worktree and did not reproduce a timeout. Current CPU samples showed 8.4–14.8% overall system usage on a 32-logical-CPU i9-13980HX, while T05 Python process CPU/wall ratios were about 0.96. The episode is CPU-bound; there is no present evidence of a blocked wait or severe aggregate CPU contention. Per-core frequency, thermal throttling, and the historical run’s contention are not recorded.

`RuntimeSession.wait()` joins the simulation thread with a timeout (`session.py:239`); `close()` cancels and joins it (`session.py:246`). The historical timeout continued making physics progress until cleanup recorded `cancelled`. Current completed traces show terminal `finished` and stopped threads. Evidence supports a physics throughput timeout, not controller stall, navigation failure, replay overhead, or deadlock. A historical `cancelled` status is timeout cleanup, not a collision or terminal-physics result.

## 5. Differential geometry and correctness

The final corrected differential run (`trace_id=ef12f30bcad744d8ac02b8cc34d37e0f`) compared the indexed path with exhaustive geometry on a synthetic 64-patch sloped road and 540 poses. It included exact seams and tiny seam offsets, road-edge grazing, overhang, yaw rotation, pitch, roll, and thresholds around the nearest SAT gap. Results:

- 276,480 exhaustive SAT patch comparisons; 0 candidate-inclusion misses.
- 0 swept-clearance verdict mismatches.
- 540 finite tire-coverage states; 540 successful indexed/exhaustive equalities, 0 exceptions, 0 mismatches.
- Broad-phase retained 27.8% of synthetic road candidates.

The final fixture used one continuous plane across the seams, so it does not establish completeness for arbitrary discontinuous tile slopes. The atlas separately rejects height-inconsistent overlapping seams. Non-finite or unbounded queries fall back to all candidates, preserving fail-closed handling. The existing bounded-clearance tests also passed. An earlier draft of the differential harness created invalid tilted wheel-coverage states and raised 180 comparison exceptions; the harness was corrected to align the body with the sloped road and the final 540-state run completed with no exceptions. No exception was dropped from the final run.

These tests found no false-negative contact in the sampled seam, slope, overhang, rotated, or grazing cases. They are broad adversarial samples, not a mathematical proof over all geometry inputs. The candidate index is conservative in tested cases and exact SAT/coverage routines remain the final verdict path.

## 6. Initial audit findings: Physics, PPO, factory, and source provenance (updated in §10)

At the initial audit point, the safe next performance candidate was an episode-scoped cache/reuse of the validated atlas and bounded patch index. It should be owned by the session/physics integration, tied to the exact immutable surface tuple, seam tolerance, and body geometry, and invalidated whenever one changes. Keep construction validation and the existing all-candidate fallback for invalid/unbounded queries. Avoid a global cache whose key or lifetime could reuse an atlas across different environments. The initial measured upper bound was about 9.3 s per 477-step T05 episode before cache-management overhead. Section 10 records the implementation and acceptance.

Changing energy tolerance, reducing refinement depth, skipping RK stages, weakening SAT, or changing polygon coverage is a risky semantic change and is not recommended. More workers would improve aggregate episode throughput at most; this sequential simulation is already consuming roughly one CPU core, so workers would not shorten the latency of an individual episode.

The optimization changes physics source bytes. The frozen WP3 registry expects `spatial_contact_dynamics.py` SHA-256 `62895c5ddf042135dc9e10e8d143f3d696cbf7b0d84d8fa0abeb7d2b97a3d32c`; the final worktree file is `3f33b18f1d69e6fd4f7599bbe5b4d30c396231f1e23ecf10bed228dc8dc4ef8d` (initial audit worktree file: `99b7b16d915d1919dc6f2a0aa9eedfaba79d2d6facc1e8353e4243b064172213`; HEAD file: `cc9842f91b09ee79ab03509a5adbcd299637b0ac7a27e9ee09c03fb851cf8c40`). The WP3 source-bound set also includes the modified `swept_surface_clearance.py`; the modified `finite_tire_footprint_v4.py` is not listed in that fixed source bundle. No acceptance record or source binding was changed. The final bytes require physics-owner source-bound review before future scientific use.

The cache and query changes are intended to preserve collision, contact, reward, observation, and action semantics, so they do not by themselves require PPO retraining. They also do not make existing scientific provenance apply to new source bytes. The admission tests remain tied to the canonical typed configuration contract; adding `research_admission_required` would create a separate public contract and could introduce an unsafe override, so it should remain absent unless Băng’s canonical API is formally revised.

## 7. Verification run

Focused regression command used the repository `.venv` and a namespace-package shim for the local `tests` directory:

```text
.\.venv\Scripts\python.exe -c "import sys,types,pytest; package=types.ModuleType('tests'); package.__path__=['D:/Projects/F1-RL/tests']; sys.modules['tests']=package; raise SystemExit(pytest.main(['-q','tests/sandbox/test_bounded_patch_clearance.py','tests/sandbox/test_spatial_contact_dynamics.py','tests/sandbox/test_spatial_atlas_lifecycle.py','tests/sandbox/test_spatial_compliant_lifecycle.py','tests/sandbox/test_spatial_rotating_lifecycle.py','tests/sandbox/test_unilateral_spatial_contact.py','tests/learning/training/test_env_factory.py']))"
```

Result: **76 passed in 16.53 s**.

TRAIN baseline:

```text
.\.venv\Scripts\python.exe -m pytest -q --tb=short tests/runtime/test_auto_drive_integration.py::test_right_hand_auto_drive_finishes_slalom_without_collision
```

Result: **PASS in 16.68 s**, 318 steps, all 3 checkpoints completed, route progress 1.0, collision count 0. No full regression or scientific evaluation was run.

## 8. Initial audit plan and acceptance status (superseded by §10)

1. **Preserve evidence while optimizing:** implement an episode-owned cache for the atlas and patch index only after source-bound review. Keep seam validation, conservative candidates, exact final SAT/coverage, and fail-closed fallback unchanged.
2. **Prove equivalence:** rerun the 276,480-query differential suite plus exact seam, near-grazing, rotated, slope, and invalid-query cases; require zero candidate misses, zero verdict/coverage mismatches, and no new exceptions.
3. **Prove runtime benefit:** benchmark whole serial episodes and early/middle/late steps with and without instrumentation. Retain the 120 s timeout. Require all five exact Auto Drive/T05 IDs and TRAIN Slalom to pass, with stable source fingerprints and per-run CPU/system telemetry.
4. **Keep the factory API canonical:** retain the typed, derived, fail-closed admission guard. The 9 current factory tests must pass; do not restore the stale marker tests unless the canonical handoff changes.
5. **Close scientific provenance separately:** physics owner reviews all changed dependency files and the appropriate registered bundle before any scientific use. No PPO retraining is indicated by a semantics-preserving cache alone.

**Initial audit verdict at the time (superseded by §10): `NEED_MORE_EVIDENCE`.** All five exact tests passed in that audit run, and the broad-phase had differential evidence. The earlier archive showed materially different exact SAT-kernel call counts between the 120 s failure and 85 s success but lacked source hashes and historical machine telemetry. The user-reported 3/5 run and unusually high step-61 timings were not reproduced then. Section 10 records the later implementation, controlled timeout reproduction, repeated exact-node runs, and current verdict.

## 9. Follow-up hypothesis check and ordered plan (before P1–P3 implementation)

A second external trace instrumented only energy-refinement branches at steps 61, 120, 240, 360, and 460 of the exact T05 normal DEVELOPMENT test. The test passed in 75.87 s. The trace logged the existing energy terms at the branch point; it did not modify repository source or tests (`trace_id=7e8f11e9c7b44d069df05d379363f697`).

At step 120, the first rejected 6.25 ms interval had:

| Energy-ledger term | Value |
|---|---:|
| Kinetic energy change | +68.358 J |
| Gravitational potential change | +2.767 J |
| Spring energy change | -4.066 J |
| Actuator work | 176.645 J |
| Dissipation | 109.868 J |
| Resulting residual | +0.282765 J |
| Registered tolerance | 0.000219773 J |

The large work, dissipation, and mechanical-energy terms mostly cancel; the remaining sub-joule balance error is about 1,287 times the registered tolerance. Across the 48 rejected trials sampled at this step, mean absolute residual was 0.0799 J and the maximum was 0.7057 J. The same qualitative pattern appears at the other sampled steps. This confirms the immediate CPU multiplier: `advance()` computes the full five stage evaluations, finds the local balance outside tolerance, and recursively computes both half intervals. At step 120 that produced 112 `advance()` calls and 561 `stage()` evaluations, versus 16 and 81 without energy splits. Swept chassis certification is a separate per-substep path: the same step recorded 64 certificate calls, not one for each of the 112 `advance()` calls.

All five sampled steps kept the same active-wheel set `{0,1,2,3}` across the rejected RK stages. This trace did not record each stage’s patch ID or numeric coverage, so it does not establish whether smooth tire-footprint changes or a patch transition dominate the energy defect. It does rule out a change in which wheels are active as the explanation for these sampled splits. The residual is the discrepancy among RK4 mechanical-energy change, actuator work, and dissipative work; changing its tolerance would change the registered solver contract and is not recommended.

Source review confirms two extra details about avoidable work:

- `_single_polygon_coverage()` recalculates polygon edge deltas and lengths for every candidate wheel-patch query (`finite_tire_footprint_v4.py:29–56`). The edge geometry is static for the episode; only the wheel point and resulting signed distances vary.
- `candidate_indices_near_xz()` merges a static `always_footprint_indices` tuple with one static cell tuple through a new set and sort on every call (`swept_surface_clearance.py:121–136`). The archived full episode recorded 485,108 such tire candidate queries. A pre-sorted per-cell union can preserve candidate order while removing repeated set/sort work. The road query similarly computes body axes for its bounds, then `has_separation_at_most()` computes the same axes again (`swept_surface_clearance.py:38–85`).

The completed T05 trace recorded 121,277 finite-tire stage calls (about 14.5 s inclusive), 15,896 swept-road certificates (about 14.5 s inclusive), 63,584 patch separation queries (about 13.3 s inclusive), and 5,503 exact OBB lower-bound calls (about 0.38 s). These timers overlap. They show that the exact SAT kernel itself is not the priority; candidate-query bookkeeping and tire coverage are better targets. The step-120 cProfile also attributed 0.174 s cumulative to finite tire coverage and 0.151 s to swept certification, out of 0.662 s for the entire profiled integrator step.

Recommended order at that point, with the split condition and tolerance left untouched:

1. **Build one prepared geometry context per `Simulation`.** It should own the validated atlas, patch index, patch lookup, and static coverage geometry for the exact track surfaces, seam tolerance, and body geometry. `Simulation` already owns the fixed `TrackWorld` and survives steps; pass this context through `step_world()` into the V4 integrator. A new simulation or changed geometry creates a new context. Preserve validation on construction and all-candidate fail-closed behavior. The likely files are `f1rl/runtime/simulation.py`, `f1rl/sandbox/world.py`, and `f1rl/sandbox/spatial_contact_dynamics.py`; the prepared geometry type belongs with the spatial support/index code.
2. **Remove repeated static and per-query work inside the stage.** Precompute polygon edge coefficients and ordered tire candidate tuples when building the context. Compute body axes once per road query and reuse them for both bounds and separation checks. Reuse wheel arms/points within a stage if the coverage and force path can share the exact same rotations and preserve arithmetic order. These changes target measured coverage and query costs without altering the energy split rule.
3. **Reuse only exactly identical recursive inputs.** When an energy split recurses into its left half, its initial `stage(y)` is identical to the parent’s already computed `a=stage(y)`. Passing that immutable result to the left child can avoid one stage evaluation per split. The step-120 case would avoid 48 of 561 stage calls (8.6% at that step). Keep all other RK stages and both children; do not extrapolate or reuse results at different states.
4. **Measure allocations before changing object construction.** The profile proves `stage()` is hot, but no allocation profile was collected. Replacing `Vec3`, `VehicleState`, `_Stage`, or audit record construction is not yet justified by allocation evidence and risks floating-point/order changes.

This section records the pre-implementation prioritization. Section 10 reports that the context and query changes were implemented and verified, while recursive stage reuse was measured and intentionally not retained because its episode-level gain was not material.

## 10. Implementation and final acceptance addendum — 2026-11-09

### Final repository state

- Branch: `main`
- HEAD: `6f0110a871087828a9c9ce16327281bafbabf4d6`
- Worktree: modified before this implementation and remains modified; all initial changes were preserved. No commit, push, merge, reset, or branch switch occurred.
- `git diff --check`: PASS.
- Final accepted phases: P1 and P2. P3 was evaluated and then removed because whole-episode benefit was below measurement noise.
- Structured benchmark, profiler, equivalence, and per-run JSONL traces are outside the repository at `D:\Projects\.archives\F1-RL-targeted-audit-20261109`.

Final source SHA-256 values:

| File | SHA-256 |
|---|---|
| `f1rl/runtime/simulation.py` | `D9C67A580537FDB445A03C8D6394D2C381D93718729472F66AB79B7A7BD55373` |
| `f1rl/sandbox/world.py` | `5D56640D7EFB2C91C5D71F101C227F4C5FBEEC549F0BD46087D8375AE92EF218` |
| `f1rl/sandbox/spatial_contact_dynamics.py` | `3F33B18F1D69E6FD4F7599BBE5B4D30C396231F1E23ECF10BED228DC8DC4EF8D` |
| `f1rl/sandbox/swept_surface_clearance.py` | `1E1440A6AD9778B9435CF3C3E4E21C567F97A5EB0EDFC996BB3BE4B104A440FB` |
| `f1rl/sandbox/finite_tire_footprint_v4.py` | `9E50E54FF3F4DF4950EEA4394D589670B60FA9FA336E47C48DE7184432EFFCF0` |

Files with this task’s retained implementation or verification changes:

- `f1rl/runtime/simulation.py`: own/reuse a prepared geometry context for each Simulation and rebuild when its exact surface/config/body key changes.
- `f1rl/sandbox/world.py`: pass the context only to the registered Physics V4 path.
- `f1rl/sandbox/spatial_contact_dynamics.py`: validate/use the prepared V4 atlas and patch data, retaining the existing fail-closed path.
- `f1rl/sandbox/swept_surface_clearance.py`: precompute bounded-query data, per-cell ordered tire candidates, polygon edge coefficients, obstacle axes/extents, reuse body axes inside the same query, and expose cached lookup maps as read-only views.
- `f1rl/sandbox/finite_tire_footprint_v4.py`: consume precomputed edge coefficients while preserving coverage arithmetic and patch order.
- `tests/runtime/test_spatial_geometry_context.py`: cache match, reuse, invalidation, invalid-atlas, and read-only index-map cases.
- `tests/sandbox/test_bounded_patch_clearance.py`: P2 edge/candidate reference checks. The existing broad-phase tests in this already-dirty file were preserved.
- `PHYSICS-V4-INDEPENDENT-PERFORMANCE-REAUDIT-20261109.md`: this addendum.

`tests/learning/training/test_env_factory.py` was already modified when this task began; its stale marker-test removal was not changed here. The other targeted-audit reports were also already untracked and were left untouched.

### Phase results

**P1 — prepared geometry context.** The focused suite passed 79 tests in 22.27 s, including three new context tests. The tests observed one context build across repeated steps, a rebuild after the track surface changed, and unavailable context for invalid atlas data. The final context also owns a frozen index whose maps are read-only mapping proxies. T05 normal took 79.188 s wall / 76.438 s reported Python child CPU, then 78.174 s / 74.266 s. With the same P1 source, an external diagnostic disabled only Simulation context reuse; T05 normal passed in 93.264 s / 89.313 s. This confirms the cache removes repeated static construction, though per-run timings vary.

**P2 — tire coverage and spatial query.** The differential harness compared 276,480 SAT cases over seam, grazing, overhang, yaw/pitch/roll geometry, and 540 tire-coverage states. It recorded zero candidate-inclusion misses, zero verdict/coverage mismatches, and zero exceptions. The road query retained about 27.8% of candidates in that synthetic fixture. The focused suite passed 81 tests in 26.42 s, and the final suite with immutable maps passed 81 tests in 17.16 s. Corrected per-step profiling of the final source sampled steps 1, 61, 120, 180, 240, 300, 360, 420, and 460: road queries made 64–256 candidate calls per step, kept an average of 6–17 candidates from 142 patches (4–12%), and made zero exact OBB SAT calls at those samples because the conservative body-axis gap rejected the candidates first. Tire lookup made 324–2,244 candidate calls, averaging about 3–8 candidates each. Step 120 retained the original 112 `advance()` and 561 `stage()` calls. The profiled T05 normal episode passed in 65.699 s wall / 64.031 s process CPU. In a runtime control with P2 query/coverage paths disabled and P1-style dynamic sorting, edge recomputation, and candidate SAT restored, the exact normal T05 node timed out at 120.312 s with 116.859 s process CPU. That control reproduced the timeout as CPU work, not a stuck session thread. Final repeated P1+P2 runs all passed (table below).

**P3 — recursive initial-stage reuse.** A temporary P3 implementation reused the immutable `stage(y)` only for an energy-refinement left child with identical state object, identical start time, and the same integration-call context token. At T05 step 120 it preserved exact equality for every `SpatialGroundResult` field against a saved P2 implementation; `advance()` stayed at 112 calls and substeps stayed at 64, while `stage()` fell from 561 to 513 calls. However, full-episode results did not establish a repeatable practical gain: P3 completed in 62.108 s; a P3 run with one differential reference step took 63.03 s; the P2-reference episode with matching cache completed in 63.28 s. Final P1+P2 T05-normal runs took 61.961 s and 69.032 s. Since the stage-count reduction did not translate into a measurable episode improvement, the P3 source and test delta were removed. The final source hash for `spatial_contact_dynamics.py` matches the captured P2 snapshot byte-for-byte.

Normal T05 phase trajectory (single-episode measurements; timing varied by run):

| Phase / mode | Measurement | Outcome |
|---|---:|---|
| P0 baseline from initial audit | 73.623 s wall / 70.703 s CPU | PASS |
| P1 cache enabled, run 1 | 79.188 s wall / 76.438 s CPU | PASS |
| P1 cache enabled, repeat | 78.174 s wall / 74.266 s CPU | PASS |
| P1 context reuse disabled control | 93.264 s wall / 89.313 s CPU | PASS |
| P2 phase benchmark | 87.444 s wall / 83.828 s CPU | PASS |
| P2 profile run before final read-only-map check | 77.896 s session wall / 75.172 s CPU | PASS |
| P2 final source, corrected geometry profile | 65.699 s session wall / 64.031 s CPU | PASS |
| P2 paths disabled, P1-style query control | 120.312 s wall / 116.859 s CPU | TIMEOUT |
| P3 experimental implementation | 62.108 s wall; CPU not separately captured | PASS; removed |
| P3 with one differential reference step / P2 reference episode | 63.03 s / 63.28 s wall respectively | PASS / PASS; no material delta |
| Final retained P1+P2, exact T05 normal round 1 / 2 | 61.585 s wall / 58.828 s CPU; 61.891 s / 59.016 s | PASS / PASS |

The P2-off control confirms the cost of repeated query work under the existing 120 s wait. The profile and repeated final runs show the final P1+P2 source completing below 70 s on this host; the P0/P1/P2/P3 single-run values are not a controlled cross-phase distribution, so the table should be read with the recorded run mode and CPU evidence rather than as a universal speedup percentage.

### Final five-node results

Each test ran individually and serially in two rounds, with the unchanged `session.wait(120)` contract. All ten runs passed. Reported CPU is the external runner’s Python-child CPU measurement; system CPU was sampled about every 10 seconds. Runtime varied across rounds, with system CPU samples generally below 20% and a brief 29.5% sample.

| Test ID | Round 1 wall / CPU (s) | Round 2 wall / CPU (s) |
|---|---:|---:|
| `test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]` | 16.941 / 14.891 | 15.829 / 14.531 |
| `test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]` | 16.135 / 14.797 | 16.222 / 14.672 |
| `test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve` | 44.349 / 41.453 | 43.740 / 41.453 |
| `test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]` | 61.585 / 58.828 | 61.891 / 59.016 |
| `test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]` | 62.099 / 59.094 | 68.358 / 65.516 |

The five nodes also passed together in one pytest process: **5 passed in 192.18 s**, 192.973 s outer wall / 185.672 s reported Python child CPU.

Final focused regression command (same namespace shim as §7, with the new geometry-context tests included):

```text
.\.venv\Scripts\python.exe -c "import sys,types,pytest; package=types.ModuleType('tests'); package.__path__=['D:/Projects/F1-RL/tests']; sys.modules['tests']=package; raise SystemExit(pytest.main(['-q','tests/sandbox/test_bounded_patch_clearance.py','tests/sandbox/test_spatial_contact_dynamics.py','tests/sandbox/test_spatial_atlas_lifecycle.py','tests/sandbox/test_spatial_compliant_lifecycle.py','tests/sandbox/test_spatial_rotating_lifecycle.py','tests/sandbox/test_unilateral_spatial_contact.py','tests/learning/training/test_env_factory.py','tests/runtime/test_spatial_geometry_context.py']))"
```

Result: **81 passed in 17.16 s**. TRAIN Slalom passed in 13.72 s. Structured runtime metrics recorded 318 steps, 319 frames, route progress 1.0, checkpoint IDs `checkpoint-0`, `checkpoint-1`, `checkpoint-2`, and zero collision frames. The focused command included `tests/learning/training/test_env_factory.py`; its canonical contract tests passed. No research guard or factory source was changed in this task.

### Effects, risks, and disposition

The confirmed episode cost grows at energy-refinement points: the step-120 reference had 112 `advance()` and 561 `stage()` calls. P1 removes repeated static atlas/index setup. P2 reduces repeated geometry preparation and query bookkeeping while keeping the exact tire-coverage/SAT routines as the final verdict. The disabled-P2 control’s 120 s timeout and the final repeated passes support P2 as the timeout fix on this host. The final run set does not prove identical wall time on every machine; P0/P1/P2 point timings ranged materially, and aggregate CPU can vary.

P1/P2 add new bytes to source-bound Physics V4 modules. Existing frozen evidence, source registries, research guards, rewards, observations, actions, contact thresholds, energy tolerances, and refinement rules were not modified. The broad differential sample is not a proof over every possible polygon/pose. Physics-owner source-bound review remains required before scientific use. No PPO training, HELD-OUT work, scientific evaluation, extra workers, or full regression was run. PPO retraining is not indicated by this semantics-preserving optimization.

**Verdict: `READY_FOR_REVIEW`.** P1 and P2 are retained and verified; P3 is intentionally not retained because its measured end-to-end gain was not material. No commit or push has been made.
