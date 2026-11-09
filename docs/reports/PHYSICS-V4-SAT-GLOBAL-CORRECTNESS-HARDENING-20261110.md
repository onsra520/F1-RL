# Physics V4 Global SAT Correctness Hardening — 2026-11-10

## Verdict

`READY_FOR_REVIEW`

Follow-up: a later targeted fixture found a near-parallel cross-axis gap error not covered here. This report's source hash and verdict pre-date that fix. See `PHYSICS-V4-SAT-NEAR-PARALLEL-FIX-20261110.md` for the final targeted change and verification.

The shared float-plus-Decimal SAT decision passed the independent differential set, focused collision and terminal regressions, and all five timeout-sensitive Auto Drive/T05 nodes in three final serial runs. Under low, comparable host load, the final candidate was within 1.7% wall time and 1.1% process CPU time of the measured baseline. Higher-load runs were slower and are reported separately.

This is bounded differential and regression evidence, not a formal proof over every finite IEEE-754 geometry. No scientific admission was opened.

## Repository and worktree state

- Repository: `D:\Projects\F1-RL`
- Worktree: `D:\Projects\.worktrees\F1-RL\sat-broadphase-workers`
- Branch: `codex/sat-broadphase-workers`
- Worktree HEAD: `74a5147bfc8f907670b45f5fd02315505262813d`
- `main` HEAD: `74a5147bfc8f907670b45f5fd02315505262813d`; `main` worktree is clean.
- Verification environment: Windows, Python 3.12.15, pytest 9.2.0.
- The pre-existing dirty pytest-worker changes and prior reports/design artifacts were preserved. No branch switch, commit, push, merge, or source-lock/binding edit occurred.
- The SAT audit touched the implementation and test files listed below. `pyproject.toml`, `requirements-test.txt`, `scripts/benchmark_vector_envs.py`, the two runtime test files’ worker markers, `tests/learning/training/test_vector_env_benchmark.py`, and prior worker/design reports were already dirty before this audit and were not altered by it.

## Root cause

Collision decisions were made by comparing a rounded float SAT gap directly with a physical distance threshold. The OBB kernel rotates and normalizes axes, projects centers and support radii, and subtracts nearly equal values. Near a threshold, the last few floating-point bits can put the same mathematical gap on either side of the decision boundary. A false positive can become a crash/contact; a false negative can clear a collision or terminal path.

The previously cited 60 m fixture was not an exact-oracle false negative: its independently evaluated normalized-quaternion gap was `1.0000000903712165e-9 m`, above `1e-9 m`, while the old float kernel returned `9.999999781306063e-10 m` and reported contact. That case was a float false positive hidden by an earlier broadphase, not a true exact-geometry broadphase miss. Separate randomized and targeted cases did show raw float SAT classifications in both directions.

## Numerical decision contract and implementation

`f1rl/sandbox/body_collision.py` now owns `obb_separation_at_most` as the shared threshold decision:

1. Compute the existing fast float 15-axis SAT gap.
2. Compute a conservative uncertainty band of 256 ULP at the largest absolute center, half-extent, or threshold scale.
3. Outside that band, return the float comparison unchanged.
4. Inside the band, recompute the normalized-quaternion 15-axis gap with 100-digit Decimal arithmetic and compare it with the exact Decimal representation of the float threshold.

The fallback is used only for boundary/overflow cases. The physical thresholds, SAT axes, vehicle physics coefficients, rewards, observations, actions, and terminal APIs were not changed. Invalid or non-finite threshold/geometry inputs are rejected at the shared boundary; the bounded index remains fail-closed.

Swept collision now reuses the midpoint SAT gap when checking both sampled contact and the conservative interval-motion bound. Both threshold comparisons share one precise fallback if either comparison is in the uncertainty band. This avoids repeating SAT and Decimal work for the same pose.

## Collision call-site audit

| Path | SAT use after audit |
|---|---|
| `body_collision.py` | Shared comparator owns scalar SAT threshold decisions; sweep endpoint and midpoint decisions use it. |
| `world.py` | World collision and terminal handling route through `sweep_body_against_obstacle`. |
| `physical_crash.py`, `ballistic_flight.py` | Crash and ballistic paths route through the shared swept kernel. |
| `swept_surface_clearance.py` | Road-patch broadphase remains conservative; final candidate verdict uses the shared comparator. |
| `spatial_contact_dynamics.py`, `runtime/roll_envelope_validation.py` | Road-free roll-witness checks use the shared comparator. |
| `linear_first_body_contact.py`, `ballistic_body_impact.py`, `body_impact.py` | Contact, interval-clearance, and initial-overlap decisions use the shared comparator. |
| `certified_face_impact.py`, `certified_edge_impact.py` | Witness-overlap checks use the shared comparator before certificate-specific geometry checks. |
| `euler_ode_clearance.py`, `wp7_arc_trajectory.py`, `wp7_planar_trajectory.py` | Obstacle-clearance decisions use the shared comparator. |
| `accepted_road_body_contact.py` | Removed an unused raw SAT import; this path’s remaining gap is a separate road-contact witness calculation, not an OBB SAT collision decision. |

`certified_edge_impact.py` still ranks SAT axes in float to select a unique contact feature. Its decision is guarded by a `1e-5 m` minimum dominance margin, a bounded contact-distance witness, and a finite-edge witness; it is not the collision/no-collision boolean. This separate feature-selection calculation remains a numerical review point if future contracts tighten that margin. The face certificate’s support-point/tangent checks also remain float geometry checks after shared SAT has established overlap.

## Independent differential method and results

The external deterministic harness at `D:\Projects\.staging\sat-correctness-20261110\sat_oracle_diff.py` evaluates a separate Decimal OBB oracle over all obstacles. The oracle normalizes the stored quaternion components and evaluates all 15 axes; it does not call the production float SAT or shared comparator to determine expected results. The harness compares every broadphase candidate and final any-obstacle decision.

- Seed: `0x5A71C0DE`
- 128 indexed obstacles × 128 poses; 16,384 exact obstacle-pose gap evaluations.
- 131,072 obstacle/threshold comparisons over 8 thresholds per pose: fixed `0`, `1e-9`, `0.05`, `0.8`, `5 m`, plus a target gap rounded to float and its adjacent representable values.
- Coverage: 56 randomized rotated/sloped patches, 64 seam patches, 8 cell-boundary patches, 96 targeted poses, and 16 far-negative poses.
- Expected decisions: 575 positive, 449 negative.
- Raw float SAT vs oracle: **15 false negatives, 118 false positives** at pair/threshold level. An independent `fsum` float formulation found 17/115, corroborating rounding sensitivity.
- Shared comparator vs oracle: **0 false negatives, 0 false positives**; **0 changes outside** the 256-ULP band.
- Broadphase: **0 required-candidate omissions**; 41,340 conservative extra candidates. Those are safe candidate-set extras, not final false-positive collision decisions.
- Final whole-index query: **0 false negatives, 0 false positives**; 0 counterexamples; 0 candidate-order nondeterminisms.
- Near-contact threshold observations: 355; boundary queries: 24; seam queries: 32; far-negative/random queries: 16.
- Invalid NaN, infinity, and negative clearance inputs remain fail-closed in the bounded index.
- Differential harness elapsed time: 26.56 s.

## Regression and Auto Drive/T05 results

Focused collision/terminal command used the worktree sources and a `tests` namespace shim required by this repository’s local test package layout. It selected:

```text
tests/sandbox/test_bounded_patch_clearance.py
tests/sandbox/test_body_collision.py
tests/sandbox/test_world_step.py
tests/sandbox/test_multi_surface_support.py
tests/sandbox/test_torque_free_swept_ccd.py
tests/sandbox/test_euler_ode_clearance.py
tests/research/test_wp3_linear_first_contact.py
tests/research/test_wp3_ballistic_body_impact.py
tests/research/test_wp3_body_impact_bracket.py
tests/research/test_wp3_certified_face_impact.py
tests/research/test_wp3_unique_edge_contact_v4.py
tests/research/test_wp3_physical_crash.py
tests/research/test_wp3_roll_envelope_terminal_v4.py
tests/integration/test_wp3_registered_crash_terminal_contract.py
tests/integration/test_wp3_applied_crash_end_to_end.py
tests/runtime/test_collision_terminal.py
tests/learning/evaluation/test_wp3_crash_terminal_gate.py
```

Result: **216 passed in 37.03 s**. The smaller direct SAT/collision command also passed **35/35**, and the TRAIN-asset geometry regression modules `test_wp7_arc_trajectory.py` and `test_wp7_planar_trajectory.py` passed **27/27**. These are deterministic engineering tests; no PPO policy evaluation or evidence generation was run.

The exact five Auto Drive/T05 node IDs were executed serially, in one pytest process per run, with existing `session.wait(120)` unchanged:

1. `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]`
2. `tests/runtime/test_auto_drive_integration.py::test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]`
3. `tests/runtime/test_auto_drive_integration.py::test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve`
4. `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]`
5. `tests/runtime/test_t05_hitbox_regression.py::test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]`

Final-source runs 5, 6, and 7 each passed **5/5**. Their per-test durations were below 120 s; the slowest was 73.30 s. No test timeout, thread-lifecycle hang, or worker was observed. Pytest/execnet processes were absent after each completed run.

## Performance comparison

Measurements use the same external serial runner, 1 Hz host/process resource sampler, exact five nodes, and pytest environment. CPU time is process CPU time; memory is peak process RSS. Stable comparisons below use low-load runs with similar host CPU averages.

| Run | Revision | Result | Wall (s) | Process CPU (s) | Host CPU avg / peak | Peak RSS (MB) | Trace ID |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline 1 | pre-global comparator | 5/5 | 204.371 | 198.344 | 13.01% / 21.2% | 242.8 | not retained |
| Baseline 2 | pre-global comparator | 5/5 | 196.031 | 189.828 | 13.46% / 76.3% | 242.4 | not retained |
| Candidate 5 | final source | 5/5 | 231.595 | 225.391 | 17.59% / 33.9% | 240.8 | `044c09911b5e4a9fabc974871a8467e0` |
| Candidate 6 | final source | 5/5 | 204.738 | 198.469 | 13.17% / 28.1% | 242.6 | `8374f5952ec94678a9d2b267e0e95929` |
| Candidate 7 | final source | 5/5 | 202.659 | 193.891 | 12.71% / 23.3% | 241.5 | `3b970d6bf9dc4b2daa9a5b6a1516a7d5` |

Baseline mean: **200.201 s wall / 194.086 s CPU**. The two low-load final candidate runs averaged **203.699 s wall / 196.180 s CPU**, or **+1.75% wall / +1.08% CPU**. Candidate 6 nearly matches Baseline 1 (+0.18% wall and +0.06% CPU) at comparable host CPU load. Candidate 5 was about 15.7% slower wall time while host CPU averaged 17.59%; it is recorded as a loaded run and is not used as the stable performance conclusion.

A paired 100,000-call microbenchmark sample of far-clear SAT checks measured raw float vs shared comparator at **5.547 s vs 6.719 s (+21.1%)** before removing redundant validation/tuple work, and **6.109 s vs 6.578 s (+7.7%)** after. The uncertainty band and Decimal fallback condition were unchanged. This isolates comparator overhead and is not a rollout-throughput estimate.

## Physics, terminal, and provenance impact

- Only threshold decisions inside the 256-ULP band can differ from the former float decision; the independent test set recorded no changes outside the band.
- The changed edge is collision/contact classification near a threshold, resolved against the normalized-quaternion Decimal SAT oracle. Tests covering swept contact, world collision, crash certificates, and terminal gates passed.
- Physics coefficients, energy/refinement tolerances, rewards, observation/action contracts, and configured collision thresholds are unchanged.
- SAT source files and the WP7 trajectory source files changed, so historical source hashes are no longer the hash of the candidate. Frozen source bindings, research admission guards, and historical evidence were not edited. Scientific admission remains closed; any later scientific acceptance requires a separate source-bound review.
- No PPO learning/retraining, checkpoint creation, HELD-OUT access, scientific evaluation, or full regression was run.

## Changed files

SAT implementation/call sites touched by this audit:

- `f1rl/sandbox/body_collision.py`
- `f1rl/sandbox/swept_surface_clearance.py`
- `f1rl/sandbox/linear_first_body_contact.py`
- `f1rl/sandbox/ballistic_body_impact.py`
- `f1rl/sandbox/body_impact.py`
- `f1rl/sandbox/certified_face_impact.py`
- `f1rl/sandbox/certified_edge_impact.py`
- `f1rl/sandbox/euler_ode_clearance.py`
- `f1rl/sandbox/spatial_contact_dynamics.py`
- `f1rl/sandbox/accepted_road_body_contact.py`
- `f1rl/runtime/roll_envelope_validation.py`
- `f1rl/research/wp7_arc_trajectory.py`
- `f1rl/research/wp7_planar_trajectory.py`
- `tests/sandbox/test_body_collision.py`
- `tests/sandbox/test_bounded_patch_clearance.py`
- `tests/sandbox/sat_reference.py` (independent test oracle)

The pre-existing dirty pytest-worker files and artifacts remain in the worktree and are listed in `git status`; this audit did not alter their worker configuration or benchmark behavior.

Core source SHA-256:

| File | SHA-256 |
|---|---|
| `f1rl/sandbox/body_collision.py` | `90A1F8F575C7E8FA47530D81499CB792D0553381736D5ACC6F1DBB4B76F2590F` |
| `f1rl/sandbox/swept_surface_clearance.py` | `01FA6855F54765E0ECBC8C85F6D91B3021B5A040C79ACF8FFD07997E97C14BD7` |

Final checks: `git diff --check` **PASS**; main remained clean at `74a5147bfc8f907670b45f5fd02315505262813d`; no commit, push, merge, or branch change.
