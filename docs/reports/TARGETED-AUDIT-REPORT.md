# F1-RL Targeted Audit: Seven Engineering Failures

**Date:** 2026-11-09  
**Repository:** D:\Projects\F1-RL  
**Branch / HEAD:** main / 6f0110a871087828a9c9ce16327281bafbabf4d6  
**Scope:** Five Auto Drive timeout cases and two environment-factory cases only. No source code, research guard, source hash, historical evidence, branch, commit, or push was changed.

## Executive findings

| Failures | Classification on current main | Finding |
|---|---|---|
| Five Auto Drive / T05 cases | PERFORMANCE | Sessions advance continuously but complete only 0.725–1.818 physics steps/s. Their worker consumes about 97–98% of one CPU core. The V4 swept chassis-to-road-patch certificate performs thousands of oriented-box separation checks per physics step on custom tracks with 101–142 road patches. The unchanged 120-second wait expires before replay and terminal assertions. |
| Two F1RLEnvFactory cases | TEST_OBSOLETE | The selected canonical Băng source and its registered hashes do not contain research_admission_required. The flag plus these tests appeared together in the integration consolidation and were removed when main restored the two exact handoff source files. No production caller currently sets the flag. |

The Auto Drive evidence supports throughput as the reason these tests hit their wait limit; it does not prove what every post-timeout assertion would have returned. Do not increase the test timeout to compensate. A conservative spatial broad phase for the existing road-patch certificate is the smallest promising performance fix. It must preserve the exact V4 collision and fail-closed guarantees.

## Repository state and verification scope

Initial inspection found a clean worktree on main at HEAD 6f0110a871087828a9c9ce16327281bafbabf4d6. The branch is two commits ahead of origin/main. The report is the only intended worktree addition; all harnesses, logs, and profiler outputs are outside the repository at:

D:\Projects\.archives\F1-RL-targeted-audit-20261109

Environment: Windows 11, CPython 3.12.15, 32 logical CPUs.

The five failing DEVELOPMENT fixtures were rerun serially, one session at a time, with the original 120-second wait. Each session has one structured JSONL trace ID. TRAIN-only baseline work used the registered slalom-v1 Auto Drive test. A one-step cProfile diagnostic compared that TRAIN track with the two authorized DEVELOPMENT geometries. No sessions were profiled concurrently.

The factory command was:

~~~
.venv\Scripts\python.exe -m pytest -q --tb=short tests/learning/training/test_env_factory.py::test_research_admission_marker_reaches_created_environment tests/learning/training/test_env_factory.py::test_factory_rejects_wrong_constructor_field_types
~~~

Result: **2 failed, 7 passed in 1.55s**. Both failures are the marker-specific cases described below.

The TRAIN baseline command was:

~~~
.venv\Scripts\python.exe -m pytest -q --tb=short tests/runtime/test_auto_drive_integration.py::test_right_hand_auto_drive_finishes_slalom_without_collision
~~~

Result: **1 passed in 17.23s**. It finished all 318 steps, all three checkpoints, and the route with zero collisions. This was an engineering runtime test, not PPO training or scientific evaluation.

The five timeout cases were invoked separately through the external audit runner, each with exactly one pytest node ID. No full regression was rerun. No HELD-OUT or scientific evaluation, PPO training/retraining, or training worker was run or added.

## Exact seven failed test IDs

| # | Test ID | Result / classification |
|---:|---|---|
| 1 | tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone] | Wait expired; PERFORMANCE |
| 2 | tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier] | Wait expired; PERFORMANCE |
| 3 | tests/runtime/test_auto_drive_integration.py::test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve | Wait expired before replay/lateral assertion; PERFORMANCE |
| 4 | tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0] | Wait expired; PERFORMANCE |
| 5 | tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0] | Wait expired; PERFORMANCE |
| 6 | tests/learning/training/test_env_factory.py::test_research_admission_marker_reaches_created_environment | Constructor rejects undeclared field; TEST_OBSOLETE |
| 7 | tests/learning/training/test_env_factory.py::test_factory_rejects_wrong_constructor_field_types[research_admission_required-yes] | Constructor rejects undeclared field before expected ValueError validation; TEST_OBSOLETE |

## Auto Drive reproduction and per-session measurements

All five cases call session.wait(120) or session.wait(120.0). That call returned false. The pytest failure therefore occurs before session.replay() and before the tests’ finish, collision, lateral, or hitbox assertions. The finally block then shuts down the manager; the recorded final status and result are cancelled, not a natural simulation terminal.

The custom fixtures have zero checkpoints, so checkpoint completion is 0/0 for each. Route progress fraction is the meaningful progress indicator.

| Case | Wall / process CPU | Steps / steps per wall second | Final x,z (m) | Speed (m/s) | Route progress | Natural status / collisions |
|---|---:|---:|---:|---:|---:|---|
| Cone speed-cycle | 120.352 / 116.906 s | 213 / 1.770 | 3.000, 126.281 | 3.750 | 0.62655 | No terminal before cleanup; cancelled / 0 |
| Barrier speed-cycle | 120.458 / 117.703 s | 219 / 1.818 | 3.245, 128.647 | 3.750 | 0.63853 | No terminal before cleanup; cancelled / 0 |
| T05 centerline regression | 120.058 / 117.031 s | 102 / 0.850 | -67.382, -19.997 | 5.793 | 0.19725 | No terminal before cleanup; cancelled / 0 |
| T05 normal 45 km/h | 121.104 / 118.094 s | 89 / 0.735 | -72.711, -16.747 | 4.063 | 0.17586 | No terminal before cleanup; cancelled / 0 |
| T05 racing 90 km/h | 121.325 / 118.391 s | 88 / 0.725 | -71.301, -18.259 | 4.144 | 0.18277 | No terminal before cleanup; cancelled / 0 |

Process CPU was 97.1–97.8% of wall time, consistent with a continuously busy CPU-bound simulation thread. Structured progress events arrive through each run; there is no progress plateau. The two speed-cycle runs advanced past the obstacle region to z=126–129m without a collision. The T05 runs also advanced along the authored route. These observations argue against a controller stall, deadlock, or a worker that stopped making progress. Since the wait expires, eventual completion and final assertions remain unverified.

### Inclusive timing captured in the five failing sessions

Times below are totals for the whole timed session, not per-call averages. Nested function totals overlap; do not sum columns.

| Case | F1RLEnv.step total | Simulation.step total | AutoDrive predict | Route projection | Obstacle OBB sweep | Frame model dump |
|---|---:|---:|---:|---:|---:|---:|
| Cone | 120.031 s | 117.405 s | 160.5 ms | 426.2 ms | 795.5 ms | 5.1 ms |
| Barrier | 120.126 s | 117.523 s | 165.0 ms | 430.6 ms | 823.8 ms | 5.5 ms |
| T05 centerline | 119.830 s | 117.784 s | 105.7 ms | 276.4 ms | 579.3 ms | 2.3 ms |
| T05 normal | 120.896 s | 118.931 s | 97.6 ms | 248.0 ms | 559.3 ms | 1.8 ms |
| T05 racing | 121.120 s | 119.298 s | 91.1 ms | 235.9 ms | 554.3 ms | 2.0 ms |

The controller, projection, normal obstacle sweeps, and frame model serialization each consume well under one second across a complete timed-out session. Replay serialization is not invoked in these failures because wait fails first. On the completed TRAIN slalom baseline, replay of 319 frames took 153.5 ms end-to-end; ReplayDocument.model_dump took 1.9 ms. Serialization cannot explain the observed two-minute waits.

### Physics hotspot evidence

The one-step cProfile diagnostic used TRAIN slalom-v1 (1 surface), the speed-cycle DEVELOPMENT geometry (101 surfaces), and the T05 DEVELOPMENT fixture (142 surfaces). Profiling overhead makes these absolute step durations unsuitable as throughput benchmarks; call counts and dominant functions identify the hot path:

| Geometry | One-step wall / CPU under cProfile | V4 swept patch certificate calls | OBB separation checks in certificate | Certificate cumulative time |
|---|---:|---:|---:|---:|
| TRAIN slalom-v1, 1 surface | 0.122 / 0.125 s | Not in top profile | 256 total OBB calls across other physics/collision work | Not applicable |
| DEVELOPMENT speed-cycle, 101 surfaces | 1.746 / 1.656 s | 16 | 6,512 | 1.517 s |
| DEVELOPMENT T05, 142 surfaces | 3.014 / 2.984 s | 16 | 9,136 | 2.582 s |

In f1rl/sandbox/spatial_contact_dynamics.py:814, each V4 integration interval calls certify_swept_chassis_patch_intersection_free. In f1rl/sandbox/swept_surface_clearance.py:105-143, each call constructs a bounded OBB for every authored road patch and gap(state) computes the minimum separation by running the exact OBB SAT routine against every patch. The profiled step visits 16 intervals and makes 64 gap samples; nearly every sample scans the full 101- or 142-patch list. Cost therefore grows with authored surface count inside every physics step.

Normal obstacle sweep totals in the failing sessions are below 0.83 seconds across 120 seconds. The dominant cost in the profiler is repeated full-road-patch certification inside Physics V4.

TRAIN comparison: the registered slalom has one surface, four obstacles, and three checkpoints. Its uninstrumented focused test finished in 17.23 seconds (~18.5 steps/s). The separate cProfile+wrapper run finished 318 steps in 48.456 seconds (6.563 steps/s under profiler overhead), all checkpoints, route progress 1.0, zero collisions. DEVELOPMENT timeout rates are roughly 10–25 times lower than the uninstrumented TRAIN test rate; track geometry differs, so treat that ratio as context, not a controlled benchmark.

### Diagnosis and minimum performance fix proposal

Classification for IDs 1–5: **PERFORMANCE**, with the note that later correctness assertions are still pending. No evidence supports a real controller, collision, checkpoint, or thread-lifecycle failure from the partial runs. These fixtures define no checkpoints. They are CPU-bound because Physics V4 repeats all-patch SAT work over heavily tessellated custom road geometry.

Proposed minimum fix: add a conservative broad phase to the existing swept chassis-to-road-patch certificate, with reusable static patch bounds/OBBs and a spatial candidate lookup. Keep the existing exact OBB separation routine for all candidates; every omitted patch must have a certified lower bound proving it cannot overlap or violate the current clearance bound. Unknown, grazing, or unresolved intervals must continue to fail closed. First benchmark this on the 101- and 142-patch fixtures.

Likely source files for a reviewed implementation are f1rl/sandbox/swept_surface_clearance.py and the atlas construction/consumer in f1rl/sandbox/multi_surface_support.py and f1rl/sandbox/spatial_contact_dynamics.py. Do not change Physics v4 constants, substeps, forces, contact policy, collision tolerances, route geometry, rewards, Auto Drive behavior, or PPO. A broad phase must not turn unresolved geometry into a pass or make the certificate less conservative.

Workers are not recommended for this issue. A worker would not remove the serial per-step all-patch scan that prevents each session from reaching its terminal assertions. Separate processes could increase aggregate throughput for independent episodes on this 32-logical-CPU host, but would not reduce the CPU work or latency of an individual timed test. Do not add runtime/training workers for this fix.

## Environment-factory contract decision

The focused run failed only for the two tests that mention research_admission_required. Dataclasses.replace and F1RLEnvFactory(**values) raise TypeError because the field does not exist in current main; the second test expects ValueError for the string "yes", but validation is never reached.

The history resolves the provenance question:

- The latest fetched canonical Băng handoff is origin/handoff/physics-v4-khanh-bang-20261106 at c70d1c9c4a222c5378debe90facac3f634b2e377. It has no research_admission_required parameter or factory field.
- Local handoff ref handoff/physics-v4-khanh-bang-20261106 at 1de279c37e62a27531d661480d20fa6e095c4e84 also lacks it.
- Commit 6cdead0 added the field, strict bool validation, factory-to-environment propagation, and these two tests together. Commit ac7bfc7 then restored f1rl/learning/training/env_factory.py and f1rl/learning/gym_environment.py byte-for-byte from Băng’s canonical handoff. The two tests remained.
- No production call site in the current f1rl tree sets research_admission_required. Existing admission consumers use the computed requires_scientific_admission property.

Therefore, against the selected canonical source contract on main, classify IDs 6–7 as **TEST_OBSOLETE**, not a missing Băng handoff field. They expose a leftover integration expectation after source restoration. The minimal recommended correction is to remove or rewrite the two marker-specific test expectations so they exercise the canonical computed fail-closed admission conditions instead. Do not change production source merely to satisfy these two tests unless the owner elects to adopt the integration-only API extension.

If the owner elects to retain the extension, the implementation in both f1rl/learning/training/env_factory.py and f1rl/learning/gym_environment.py must be restored end-to-end: default false; validate with type(value) is bool at both constructors; OR the flag into the factory’s admission property, the environment’s early research validation, and its stored requires_scientific_admission result; forward the value through factory.create(). This makes true conservative and fail-closed. It is then a deliberate API contract change, not a harmless test repair.

Those two files are source-bound. Current bytes match their WP7 registration hashes exactly: env_factory.py d439369b6a7c8153f91eb3f9e207b7e36080c0590a8783fcc0cd28b2e8bc48f1 and gym_environment.py 9bf2be0531ea23297e84ccf2a63c97b4338e288dbca45251283ab7c3ff6563db. The paired-v2 Git blob lock records their CRLF-to-LF relation. Changing the files would invalidate those registered source identities and requires an owner-approved new source registration/revision; do not rewrite historical evidence or existing hashes to make a gate pass. Updating only the obsolete tests leaves both source bindings and scientific provenance unchanged.

## Fix order and acceptance criteria

1. **Performance, Physics V4 road-patch query:** benchmark a conservative spatial broad phase on the exact 101- and 142-surface geometries. Acceptance: material measured reduction in per-step CPU and 120-second tests complete naturally; unchanged V4 contact/collision outcomes, no new fail-open path, and parity coverage for seams, overhangs, grazing contact, and unresolved intervals.
2. **Reproduce the five exact Auto Drive cases serially after the approved optimization:** retain the 120-second wait. All five must reach natural finished status before replay assertions run. Cone/barrier tests must confirm speed and braking/recovery conditions; T05 centerline must pass its lateral bound; both hitbox cases must finish with zero collision and minimum SAT separation above 0.2m.
3. **Factory tests:** preferred canonical path is to remove/rewrite the two obsolete marker tests and run the focused factory tests. If the marker is instead adopted, restore the strict fail-closed API and obtain an approved new source-bound registration first; test true propagation, default-false compatibility, and invalid bool values at both boundaries.
4. **Focused verification only:** rerun affected sandbox certificate tests, factory tests, the five exact Auto Drive nodes serially, and the registered TRAIN slalom control. No full regression is needed for the initial diagnosis; wider acceptance can be scheduled after focused checks pass.

**PPO retraining:** not required. The diagnosed fix is a geometry query optimization intended to preserve Physics V4 outputs, and the factory test decision does not alter a model or training recipe. Revalidate physical parity before considering any later change to V4 behavior; retraining is not a remedy for this throughput or test-contract mismatch.

## Diagnostic artifacts and trace IDs

All artifacts are external to the repository, under D:\Projects\.archives\F1-RL-targeted-audit-20261109. Each failed-session JSONL contains structured session_start, progress, and test_end records with its trace_id; each standalone one-step profile has profile_start/profile_complete events with one trace ID.

| Operation | Trace ID | Artifact |
|---|---|---|
| Cone speed-cycle | 2dc27e2c5fe7466eae7b29ec25520cc7 | auto-cone.jsonl |
| Barrier speed-cycle | 2175379ffa644d0a90727fabc903fa8c | auto-barrier.jsonl |
| T05 centerline | 6de465aa5f6445c6a9273345f7f36fe1 | auto-t05-centerline.jsonl |
| T05 normal 45 | 0b3101eea42d4622af6c66b88e7a0672 | t05-normal.jsonl |
| T05 racing 90 | faeb6268d0b14b9d8c719d1a1aab59bc | t05-racing.jsonl |
| TRAIN slalom profile | b8ed70559b3849b88d1d5752d19684a5 | train-slalom-profile-2.jsonl / train-slalom-2.pstats |
| One-step TRAIN slalom profile | 7c4d3ac2e78a44adbf27c12d8de7495b | one_step_profiles.json |
| One-step DEVELOPMENT speed-cycle profile | e46563d63f6243ab9f27bebdee728c33 | one_step_profiles.json |
| One-step DEVELOPMENT T05 profile | 1a992ffdef7a4a34807a26812bb8b91c | one_step_profiles.json |

SHA-256 digests for the five session logs: cone 683a8df3617b1a98f8c07cfea9917882516524fcaa40a86957fc9a5ff824a150; barrier 7c5979f97d85d1e9707a1f5cc480fdb7d8da280038f17820bfa1fc64cd4b30fb; T05 centerline 4c85efa9f9c698f83dcc544152dece806bb48776d980e7699f4f5083f58e8b74; T05 normal 7bdb47abba455bc41d6627a9e6632df9923c0d93b9c2d91e4205f3654e869bec; T05 racing dbd8220e0718b520819bd13b753983f78b4d48fd884337e5258aea1624d30d27.

The profiler JSON and scripts are reproducibility artifacts only; none are tracked or imported by project source. Additional SHA-256 values: one_step_profiles.json b4b7083a33c925640833c2845acb97efcd03eb88947462882edf4d9b087ed244; one_step_profiles.jsonl fcabb529543706f801b8b20c61a7855cefd12d555a4da9f26776d003c54417e6; TRAIN profile log be5438bf1dbc9992662982998a875e4df9700d3911838b6651c2cd105f7e9771; TRAIN cProfile data 8f693fce028642249766265368d493c6b055e0529e80fe54566937fdd25c94db. The final worktree check should show this report as the only untracked file, with branch and HEAD unchanged.

