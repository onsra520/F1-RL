# WP3 TRAIN Evaluation: verified road-edge and obstacle-contact blocker

Date: 2026-10-23. Branch: `fix/wp3-evaluation-physics-contract`.

**Gate: BLOCKED for a new registered full 5,760-episode evaluation.** This investigation is a read-only scratch replay and synthetic domain regression, **not** a replacement evaluation, physics correction, or scientific acceptance. The trained model weights, frozen source `fe83f51`, all original evidence from `handoff/wp3-khanh-bang-20261021/evaluation/issue51-v1` and the sealed HELD-OUT set remain unchanged.

## Confirmed, replayed failures

| Frozen model / scenario / seed row | Last certified state or witnessed sweep | Actual failure |
|---|---|---|
| C-seed0 / Right Turn / nominal / episode 0 | Last certified tick 216: vehicle center (0.51865, 0.50189, 67.58532), speed 3.422 m/s, route lateral distance 0.519 m, 4 m hard-route boundary | `SuspensionResolutionError`, `unsupported_contact` |
| C-seed0 / Slalom / nominal / episode 2 | Last certified tick 281: vehicle center (-2.06551, 0.50081, 95.60177). Subsequent sweep witnesses body contact at approximately (-1.92220, 0.50151, 95.81488) against the obstacle OBB centered (-3.5,1.0,100.0), half-extents (2,1,2). | `FlightPhysicsError`, `verified_obstacle_contact` |
| A-seed1 / Ramp / randomized TRAIN / episode 17 | Last certified tick 92: (0.31129, 0.50919, 48.83752), speed 6.033 m/s, route lateral distance 0.311 m. | `SuspensionResolutionError`, `spatial wheel support/travel missing`, unclassified/uncertified |

**Detailed support witness at Right Turn failure:** under the registered road atlas, the integrator's attempted body/support sweep around x≈0.49965, z≈67.932 fails `invalid endpoint body or wheel support`, while all four wheel references still have patch contacts (road-entry; some also road-exit). The two planar patches meet at the right-angle transition. A nearby synthetic supported pose with yaw=0.08 rad certifies `clear`; another with yaw=0.10 rad retains **four valid wheel contacts**, but its projected body extends outside the union by approximately 0.0007114 m² and returns `outside`. This isolates a real body-overhang certification boundary, not a claim that the car center crossed the 4 m race corridor.

**Prohibited shortcut experimentally rejected:** In a disposable scratch-only monkeypatch, ignoring the *body* footprint uncovered-area check triggered 77 overhang bypasses, yet the same C-seed0 replay later failed with `spatial wheel support/travel missing`. The road-edge departure eventually loses necessary wheel support. It is therefore unsafe and ineffective to delete the verifier, globally relax clearance, count the unsupported step as simulated success, or substitute centerline deviation for wheel/rigid-body support.

**Slalom witness:** a synthetic sample from the observed flight obstacle sweep returns `SweptCollisionResult(contact=True, uncertain=False, checks=2)`. The collision is real in the SAT geometry; the unmodeled **event-time impact impulse/score** is the missing physical contract, not a false alarm. No penalty or valid collision is fabricated.

## Contract gaps requiring a new explicitly versioned physics profile

1. **Road-edge lifecycle:** Implement partial-wheel release at the bounded road seam, allow physically supported body overhang while enforcing body/road nonpenetration, and transition to an explicitly integrated airborne state once last contact releases. Include spatial linear/angular impulse and energy checks, event times, wheel-travel limits and deterministic fail-closed bounds. Existing `spatial_rotating_compliant_lifecycle_v3` documents that road-edge departure is unsupported; do not silently reinterpret it.
2. **Airborne obstacle impact:** Register a continuous, bounded first-contact time; enforce rigid-body/obstacle nonpenetration and an impact response (impulse/restitution and associated energy balance), or a separately approved absorbing terminal crash **at certified impact time**. Preserve race-scoring and reward-prefix semantics. Do not mark an unmodeled collision as a normal valid episode.
3. **Physics/profile registration:** New versioned simulation contract, source commit, seed roster, preflight and test/evidence identity are required before any revised development evaluation; the frozen registered `WP3-TRAIN-EVALUATION.json` remains `PASS_WITH_INVALID_CELLS` forever. Do not quietly rerun invalid cells in that attempt.
4. **Validation gate:** synthetic edge/collision oracles; no false clear or tunneling; finite wheel/energy/momentum budgets; continuity at phase changes; fixed paired training seed evaluation replays for C-seed0 Right Turn and Slalom plus A-seed1 Ramp; then complete 96-cell (5,760-episode) **new** evaluation in its own root, with zero invalid technical cells required before full A/B/C/D ranking.

## Completed checks for this investigation

- `tests/research/test_wp3_physics_domain_regression.py`: 3 minimal geometry/collision regressions, confirming body-overhang vs wheel support and a witnessed Slalom obstacle collision.
- New tests plus previously validated terminal contract tests: **14 passed**.
- Existing sandbox/contact/body/flight/integration subset: **95 passed**.
- The original initial bugfix branch passed **160 tests** independently.
- All three original frozen-seed reproduction attempts were run; no model retrained.
- A full 5,760-episode Eval was **not** launched, because its immutable evaluator physics precondition is not met; doing so would reproduce invalid cells rather than validate a corrected simulator.

This report is an engineering diagnosis, **not** an assertion that the user-requested physical fixes have been completed.
