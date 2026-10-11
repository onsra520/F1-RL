# WP7 TRAIN-only dynamic route diagnostic (2026-10-11)

**Scope:** Fresh recovery branch `recovery/wp7-rebuild`. This is a deterministic, deliberately simple steering heuristic exercised against the real candidate Gym physics. **It is not PPO training, scientific pilot validation, or held-out evaluation.** No final scientific seeds used.

## Reproducible diagnostic setup

- Source: `RecoveryCandidate.from_yaml` and `build_training_scenarios`, all four TRAIN tracks only.
- Controller: steer toward the current authored endpoint using orientation-local signed XZ angle, with `steering = clamp(gain × angle / 0.5, -1, 1)`; constant policy throttle. Four sampled parameter pairs: `(gain, throttle) = (1,1), (2,1), (2,0.4), (4,0.4)`.
- Maximum 500 simulation steps per rollout; accept success only when the episode reports `finished=True` and three ordered checkpoints on slalom. Track terminal flags, offtrack, obstacle events.
- Deterministic diagnostic code is recorded in `tests/integration/test_wp7_slalom_dynamic_route_reconstruction.py`. This test uses internal Gym state and is intentionally labeled *non-learned diagnostic*.

## Baseline findings at HEAD 5204427

| Track | Heuristic finished | Notes |
| --- | ---: | --- |
| straight-sprint-v1 | 4/4 | 95 or 203 steps, depending on throttle |
| right-turn-v1 | 4/4 | 123 or 274 steps; 90-degree waypoint corner can be rounded under current 6m offtrack tolerance |
| ramp-v1 | 4/4 | 134 or 386 steps |
| slalom-v1 | 0/4 | Missed third checkpoint after abrupt `[3,0,105] → [-3,0,110]`; turned to chase a passed endpoint, ran off-track at ~10–10.5m lateral distance |

Observation: the **right-turn warning is not a proof of impossibility**: this simple controller finished. It still needs a full-width/swept-path assessment, not only the thin-point vehicle model.

## TRAIN navigation repair

Tested five initial waypoint proposals plus wider-clearance refinements **in memory first**, without altering any track world geometry. Proposed route F smooths and advances the final lane change:

- Before: `[-3,0,92] → [3,0,105] → [-3,0,110] → [0,0,124]`.
- After: `[-3,0,92] → [0,0,96] → [1.5,0,102] → [-3,0,110] → [0,0,124]`.
- Expected result with four fixed heuristic configurations: **4/4 finish**, all **3 checkpoints** in order, 0 `ObstacleHit` events, no off-track truncations; highest sampled distance from route ~**2.87m**.
- Maximum authored heading discontinuity ~**43.4 degrees** (old route ~75 degrees). Sampled centerline-to-obstacle-axis-aligned-box clearance improves from ~**1.15m** to ~**1.94m**.
- Original `f1rl-track-world-v1` **slalom world fingerprint remains** `6071829f2eb790ada04c5e59b22efb205e3a20d59fbe670e68fb78ed3eb304e0`. No obstacle, surface, checkpoint, finish, spawn, or geometry changed. All six world fingerprints must remain unchanged.
- The new slalom **content fingerprint** is `e2db2953259ebb465246d7d9bfaa7cedbfc2e8b82c389f4eef655a5f04cfd959`; the manifest entry was updated from loaded, canonical v2 content, never guessed.

## Remaining scientific gates

- **Swept-volume clearance is NOT certified.** A 1.94m *point-to-box* gap may be inadequate for a vehicle footprint with nonzero width; the simulation's obstacle event reports do not prove body clearance.
- Controller completion is **not** evidence that a PPO policy can learn the course at 200k. No calibration/confirmation weights or pilot milestone outcomes are available.
- The off-track limit `6m` permits some corner cutting; true feasibility and reward exploit resistance still require additional targeted tests.
- The TRAIN route's content fingerprint changed. Any previous model/provenance report tied to the prior navigation is **not compatible** with this version; fresh training is mandatory.
- HELD-OUT worlds/routes and `main` remain untouched. Scientific freeze/WP8 remain blocked.
