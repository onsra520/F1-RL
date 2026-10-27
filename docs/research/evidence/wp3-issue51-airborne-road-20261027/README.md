# WP3 Issue #51 — Airborne bounded-road first contact (27 Oct 2026)

## Scope and original failure

- Engineering fix branch: `fix/wp3-road-impact-lifecycle-20261027`.
- Source HEAD: `6c479540134d98d7d27331879c926c8ba82c2965`, after predecessor `fc187879d96d95303308b3607a102049a56cd783`.
- Historical immutable evaluation: `WP3-OWNER-TRAINDEV-20261026-FULL-05`, **5,752 valid, 1 invalid, 7 unattempted, 95/96 cells**; exit code 2. Do not revise or reopen it.
- Exact invalid: **C-seed1 / right-turn-v1 / randomized_train / episode 52**, error `PhysicsTransitionError: unregistered road impact lifecycle`.
- Reproduced original error with original TRAIN reset/policy seed pair, original frozen PPO ZIP and exact ground-truth manifest, **0/1 valid** (`R5-BASELINE-REPLAY-RESULTS.jsonl`).
- Independent scientific review remains blocked. No HELDOUT use, retraining, FULL-06, new model ranking or scientific claim.

## Root cause and source-certified correction

The v4 RK4 source had already witnessed rigid chassis intersection with an authored bounded road patch, but **both pre/post contact `VehicleState.grounded` were False** after a within-tick certified takeoff. The old validator only recognized `grounded_applied_road_crash`, so the real airborne chassis contact became an invalid episode. Misclassifying the impact as grounded or relaxing the validator would have been incorrect.

The correction registers **`wp3_airborne_applied_road_crash_v4_dev`** as a distinct physics + evaluator-scoring source. It:

1. Retains the true airborne pre- and post-impulse flags, accepted `FlightState('airborne')`, and complete earlier takeoff/ballistic phase audit.
2. Proves the source's gravity-dependent position and velocity endpoints before permitting contact; incorrect gravity, source kinematics or phase are rejected.
3. Searches for first bounded road OBB contact on the **exact parabolic ballistic center path + source SLERP orientation**, with conservative displacement bounds and certified contact-time bracket; no linearly interpolated ballistic-center false witness.
4. Applies a feature-witnessed unilateral rigid normal impulse, checks actual linear/angular momentum and nonincreasing rigid kinetic energy, independently replays the source and impulse, and verifies the time-limited reward/gate-event prefix.
5. Exposes the separate new source version in the engineering registered crash contract. No scientific signature is invented.

## Exact-source event certificate

The precise source and impulse data are preserved without rounding in `ROAD-IMPACT-CERTIFICATE.jsonl`, recorded **only after** the independent `PhysicsConfig.validate_transition` passed.

| Quantity | Observed |
|---|---:|
| Tick-local last proven clear time | 0.07456512451171875 s |
| First contact time | 0.07456588745117186 s |
| First-contact time bracket | 7.62939453113898e-7 s |
| Source airborne integration leaf | 0.00625 s |
| Gravity | 9.81 m/s² |
| New certified ballistic Y position error | -2.77555756156289e-17 m |
| Previous linear-chord Y error | -1.23838974017687e-5 m |
| Applied impulse (X,Y,Z) N*s | (-210.17643651, 325.56790830, -1.3429684e-5) |
| Applied angular impulse (X,Y,Z) N*m*s | (-69.09659109, -44.60658270, 258.00705969) |
| Preimpact rigid kinetic energy | 20978.815329097783 J |
| Postimpact rigid kinetic energy | 20855.61060853759 J |
| Kinetic energy dissipated | 123.20472056019207 J |
| Historical prefix leaves before first contact | 33 |
| Accepted terminal flight phase | airborne |

## Frozen-source replay and validation

- **R5 original invalid replay: 1/1 valid `physical_crash` on new clean source**, 0 invalid.
- **R4 roll invalid replay: 1/1 valid `tilt_envelope_exit`**, 0 invalid.
- **R3 seven historical hard-stop invalid replays: 7/7 valid `physical_crash`**, 0 invalid.
- Focused regression on clean source: **208 PASS, 0 FAIL**.
- Full repository regression and final source engineering tag recorded separately with frozen stdout and receipts when complete.

This is engineering validation of the reproducible TRAIN-development transition. **Independent scientific Physics validation is NOT approved**; the previously declared scientific Evaluation gate remains blocked.

## Exclusions and safety

No FULL-06 launch, no source mutation of the historical FULL-05 branch, no rerun of all 5,760 episodes, no new PPO weights and no HELDOUT access. Keep Issue #51 open for scientific review and the future fresh evaluation.
