# Physics V4 SAT Correctness Hardening — 2026-11-10

## Verdict

`READY_FOR_REVIEW`

Scope note: this report covers the earlier bounded-patch SAT audit only. Subsequent global SAT review found a separate near-parallel cross-axis error. The current fix and verification are recorded in `PHYSICS-V4-SAT-NEAR-PARALLEL-FIX-20261110.md`; this report's verdict must not be read as covering that later finding.

The SAT candidate index and final bounded-patch query matched the independent oracle on the tested geometry set. The five timeout-sensitive Auto Drive/T05 nodes passed twice in serial runs. This is strong regression evidence, not a formal proof for every representable geometry.

## Repository state

- Worktree: `D:\Projects\.worktrees\F1-RL\sat-broadphase-workers`
- Branch: `codex/sat-broadphase-workers`
- HEAD: `74a5147bfc8f907670b45f5fd02315505262813d`
- `main` remained at the same HEAD and clean; no commit, push, or merge was made.
- Pre-existing uncommitted pytest-worker work was preserved. It includes changes in `pyproject.toml`, `requirements-test.txt`, `scripts/benchmark_vector_envs.py`, the two runtime test modules, and untracked benchmark/design artifacts. This audit did not modify worker scheduling or the PPO benchmark.

## Root cause

The quick body-axis SAT filter compared a projection gap in metres only if its separating axis had been normalized. The earlier unnormalized direction could scale the gap and make it disagree with the final double-precision SAT near `1e-9 m`. Quaternion rotations and SAT support projections also accumulate small rounding differences; the existing final double result could land on the opposite side of the unchanged threshold from the mathematical OBB projection gap.

The cited 60 m fixture does **not** establish a physical false negative against an independent oracle: its exact normalized-quaternion gap is `1.0000000903712165e-9 m`, greater than the `1e-9 m` threshold. The old exhaustive double kernel returned `9.999999781306063e-10 m`, which incorrectly classified it as contact. The quick filter could therefore hide a double-precision false positive and look like a broadphase false negative. Independent differential checks also found examples in the other direction, where the double kernel rounded a true within-threshold gap outward.

## Implementation

Changed:

- `f1rl/sandbox/swept_surface_clearance.py`
  - Keeps normalized body separating axes and uses an ULP guard before the quick rejection.
  - Enlarges the spatial query roundoff margin conservatively.
  - Uses a 100-digit Decimal SAT comparison only when the ordinary SAT result falls within 256 ULP of the requested threshold. The configured clearance threshold is unchanged.
  - Returns all candidates / a fail-closed contact result for malformed box geometry or invalid thresholds.
- `tests/sandbox/test_bounded_patch_clearance.py`
  - Adds an independent Decimal OBB-SAT oracle, threshold-boundary regressions, invalid-input checks, and deterministic differential coverage over all indexed obstacles.

The old float SAT produced both classifications around the threshold in the differential set. The correction did not change Physics coefficients, contact tolerance, reward, observation/action contracts, or terminal logic. Only SAT decisions inside the numerical uncertainty band can differ from the previous float result.

## Differential evidence

External deterministic harness: `D:\Projects\.staging\sat-correctness-20261110\sat_oracle_diff.py`; output: `D:\Projects\.staging\sat-correctness-20261110\sat-oracle-diagnostic.json`. The harness explicitly imports the package from this worktree. Its Decimal oracle normalizes the stored quaternion components and evaluates all 15 SAT axes independently of `obb_separation_lower_bound`.

- Seed: `0x5A71C0DE`
- 128 obstacles and 128 poses; 16,384 exact oracle gap evaluations.
- 131,072 all-obstacle pair/threshold comparisons across fixed thresholds (`0`, `1e-9`, `0.05`, `0.8`, `5 m`) and adjacent representable values around target gaps.
- Coverage: 56 randomized rotated/slope patches, 64 seam patches, 8 cell-boundary patches, 96 targeted poses, and 16 far-negative poses.
- Query outcomes: 575 positive and 449 negative.
- Broadphase candidate omissions: **0**. Conservative extra candidates: 41,340; these are candidate-set extras, not final collision false positives.
- Final query mismatches: **0 false negatives, 0 false positives**; counterexamples: **0**.
- Raw double SAT pair/threshold classifications against the independent oracle: 15 false negatives and 118 false positives. The ULP-triggered precise final check resolves these in the final query.
- Candidate order nondeterminism: 0. Invalid NaN/infinite/negative thresholds fail closed.
- Harness wall time: 13.865 s.

An exact representable `0.25 m` threshold case also passes at the threshold and at its immediately adjacent lower and higher floats. The 60 m fixture and a second 3 km rounding fixture verify that the double result cannot override the independent threshold classification.

## Regression results

Focused SAT/collision command:

```text
python -m pytest tests/sandbox/test_bounded_patch_clearance.py tests/sandbox/test_body_collision.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_torque_free_swept_ccd.py -q
70 passed in 7.09s
```

The five required nodes were run serially with their existing `session.wait(120)` contracts. A second full serial run was also made; both runs passed 5/5. Latest run: 5 passed in 223.62 s.

| Test node | Latest wall time | Result |
|---|---:|---|
| `test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[cone]` | 16.55 s | PASS |
| `test_auto_accelerates_brakes_and_recovers_speed_around_obstacle[barrier]` | 16.35 s | PASS |
| `test_right_hand_track_never_crosses_centerline_on_original_t05_s_curve` | 47.67 s | PASS |
| `test_t05_original_cone_hitbox_racing_line_finishes_without_contact[normal-45.0]` | 70.29 s | PASS |
| `test_t05_original_cone_hitbox_racing_line_finishes_without_contact[racing-90.0]` | 70.86 s | PASS |

`git diff --check`: PASS.

## Provenance and remaining risk

The SAT implementation source changes, so existing frozen source bindings remain untouched and must not be treated as approval for a future scientific run. Any scientific acceptance requires its separate source-bound review. No PPO learning/retraining, HELD-OUT access, scientific evaluation, or full regression was run.

The Decimal fallback is limited to an empirically conservative ULP band and the tested ranges. The oracle and adversarial set substantially reduce risk, but they do not prove a formal global floating-point error bound for all finite inputs. The fallback can add work for queries close to the threshold; this audit measured correctness and the timeout-sensitive sessions, not general rollout throughput.

## Source hashes

- `f1rl/sandbox/swept_surface_clearance.py`: `213B7476C7E9AF4AF0179BD9716AF56560C1B8089392CB31FD928A04A9FB18AD`
- `tests/sandbox/test_bounded_patch_clearance.py`: `204D3D56BD5C8695158B8447D23D0038EB508773F531D4CCBE29A2ECD4FE049B`
