# WP3 Evaluation — Terminal Contract and Physics Invalid-Cell Investigation

**Source:** `fix/wp3-evaluation-physics-contract`, based on `fix/wp3-eval-issue-51` (`428cefbd`).
**Frozen original training source:** `research/wp3-ppo-tuning-v2` (`fe83f517`).
**Original evidence:** `handoff/wp3-khanh-bang-20261021/evaluation/issue51-v1`, commit `2ae51b6`.
**Status:** Targeted evaluator correction verified; physical-domain blockers remain; **no registered Eval rerun authorized**.

## Evidence and exact failure breakdown

The previously published TRAIN evaluation processed 96 cells but completed only 31 cells. It recorded 2,153 valid episodes, 65 invalid episodes (one for each aborted cell), and 3,542 unattempted slots. None of the 65 invalid cells may be overwritten or backfilled in that registered evaluation.

| Track | SuspensionResolutionError | Terminal mismatch ValueError | FlightPhysicsError | Total invalid cells |
|---|---:|---:|---:|---:|
| straight-sprint-v1 | 1 | 9 | 0 | 10 |
| slalom-v1 | 15 | 1 | 5 | 21 |
| right-turn-v1 | 14 | 8 | 0 | 22 |
| ramp-v1 | 4 | 8 | 0 | 12 |
| **Total** | **34** | **26** | **5** | **65** |

These numbers describe the **first invalid episode per cell**, not 65 independent categories of successfully scored racing failure.

## Confirmed evaluator contract bug — corrected

- `f1rl/learning/gym_environment.py` returns `terminated=True`, `truncated=False`, `finished=False` for an owner-approved hard-route terminal failure.
- Previously, `f1rl/learning/evaluation/evaluator.py` treated **every** non-finish `terminated=True` episode as invalid, causing 26 `ValueError: terminal flags are inconsistent with unsuccessful episode` cases.
- The correction recognizes only the explicitly witnessed, registered outcome containing `research_outcome.version=wp7_episode_outcome_v1`, `kind=valid_terminal_failure`, `valid_sample is True`, `reason=hard_route_violation`, and `race.terminal_failure is True`.
- Such an episode becomes a **valid failed attempt**, with `success=False`, `truncated=False`, and `failure_reason=hard_route_violation`. Finished episodes, timeout truncations, and malformed/forged outcome metadata retain fail-closed semantics.
- Real `evaluate_episode()` owner-failure integration tests and strict false-positive checks are included in `tests/learning/evaluation/test_wp3_hard_route_terminal_eval.py`.

**Scratch replay (outside all registered Eval roots):** model `A-seed0`, `straight-sprint-v1`, `nominal`, evaluation seed row 0, final trained checkpoint. The previously invalid `ValueError` episode now returns a **valid unsuccessful** episode: `hard_route_violation=True`, `failure_reason=hard_route_violation`, `success=False`, 440 steps. No registered result was rewritten.

## Suspension exceptions — still invalid, not reclassified

The 34 suspension errors reflect contact/geometry certification failures:

- 31: `spatial body/contact sweep outside: invalid endpoint body or wheel support`.
- 2: `spatial wheel support/travel missing`.
- 1: `spatial body/contact sweep outside: witnessed unsupported body or wheel pose within motion`.

`f1rl/sandbox/multi_surface_support.py` returns `outside` when an endpoint or subdivided body/wheel pose is verifiably unsupported. `f1rl/sandbox/spatial_contact_dynamics.py` also rejects unresolvable wheel travel/support. The validation condition **must not** be relaxed simply to obtain a lower invalid count; first determine whether the correct outcome is a race-boundary terminal failure, an unscored valid crash, an impossible world transition, or a geometry/integrator defect.

**Scratch replay:** `C-seed0` / `right-turn-v1` / `nominal` / episode 0 reproduces `SuspensionResolutionError: spatial body/contact sweep outside: invalid endpoint body or wheel support` on the current code, with the exact frozen episode seed. Nothing was redrawn in the evidence directory.

**Required diagnosis before a fix:** add a read-only diagnostic capture in a separate execution branch for obstacle-free world replay: last certified state, first failing state/substep, swept wheel positions, yaw/pitch/roll, road patch ids, patch support gap, suspension travel, nearest route/hard boundary, and deterministic reproducibility. Compare against contact-domain source rules and approved vehicle geometry. Any physics correction needs its own tests and evaluation-source provenance.

## Flight/obstacle collision — requires explicit impact contract

All five `FlightPhysicsError` records occurred on `slalom-v1` and said `flight/substep body hit obstacle: no scored impact response registered`. In `f1rl/sandbox/world.py`, airborne or auto-released body contact triggers this error rather than fabricating a scored collision.

**Scratch replay:** `C-seed0` / `slalom-v1` / `nominal` / episode 2 deterministically reproduces the original `FlightPhysicsError` with the frozen seed. The existing obstruction is not an evaluator type/flag bug.

**Required design decision before a fix:** specify a versioned physical impact response and when it counts as a valid collision vs a terminal failure. Validate swept-contact time, impulse/energy consistency, reward terms and whether a truncated or terminated outcome is permitted; test airborne and auto-release conditions. Until an authorized scoring contract exists, keep this an invalid episode, not a successful rollout.

## Rules for the next evaluation

1. Freeze the historical evidence and the 12 trained checkpoint SHA-256 values unchanged; preserve `fe83f517` training provenance.
2. Use a new evaluator commit and evaluation-only output root/version; record both training and evaluation source IDs.
3. Retain the registered 4 m hard-route boundary, four-track roster, both strata, 60 episodes/cell, paired seed lists and original task. No HELD-OUT access or model retraining.
4. Run all regression gates and physics-specific replay tests **before** authorizing another full registered 5,760-episode evaluation. Never overwrite or silently redraw invalid episodes in the old run.
5. Treat 26 potential terminal-contract corrections as new evidence **only on a new evaluation run**. There is no valid A/B/C/D ranking from the historical partial evaluation.
6. Original canonical campaign recovery is still blocked on historical smoke/preflight prerequisites; scientific WP7, Gate 2 and HELD-OUT remain blocked.
