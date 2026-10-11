# WP7 C2.4 — TRAIN-only reconstructed candidate builder

**Date:** 2026-10-11. **Research state:** `candidate`, **NOT frozen**. Not a recovered copy of the lost WP7.

## Why a separate candidate config

`configs/experiments/ppo-generalization-v1.yml` is an original WP6 draft describing `rangefinders-local-navigation-v1` and `target-progress-collision-time-v1`. Those semantics are no longer literally the same after route endpoint and high-water reward reconstruction. Rather than silently rewrite or freeze the historical experiment, `configs/recovery/wp7-c2-candidate.yml` explicitly owns the new behavior contract and its proposed reward/off-track hyperparameters. The old experiment is used for the WP1B **vehicle, sim, sensors, episode and 4 TRAIN / 2 HELD-OUT static split** only.

The candidate currently declares:
- `ground-no-slip-air-ballistic-v1`
- `authored-surface-polyline-endpoint-v1`
- `rangefinders-local-route-navigation-v1`
- `route-highwater-progress-collision-time-offtrack-v1`
- `steering-throttle-brake0-v1`
- `single-lap-ordered-checkpoints-v1`

Proposed weights (not trained or approved): route progress `1.0 / meter`, collision `5.0`, time `0.1 / second`, offtrack `2.0 / step`. Offtrack lateral XZ limit `6.0 m`, grace `10 consecutive steps`. These are explicit hypotheses to validate in WP7; they are not asserted to match the lost successful candidate.

## `RecoveryCandidate.from_yaml` and `build_training_scenarios`

The strict loader rejects unknown or missing config keys, semantic identity drift, non-finite/negative parameters, `frozen` status and an experiment with nonempty final seeds. The builder checks static track hash identities, constructs `TrainingScenarioSet` only from `split.training`, attaches each asset's unique `AuthoredNavigation` and identical physical/sensor/reward/off-track contracts, and fails when a TRAIN track lacks v2 navigation.

Per scenario: action space **2**, observation space **16** (7 rangefinders plus 9 fixed navigation/velocity features), `RewardComposer(RouteProgressReward, CollisionPenalty, TimePenalty, OffTrackPenalty)`, and explicit off-track truncation. Legacy WP6 factories remain supported unchanged. Builder code never iterates `split.heldout` for environment creation or runs learned policy evaluation.

## Remaining prerequisites before a new WP7 pilot

- Independently review authored routes for obstacle-clearance and feasibility, step-wise route reward shortcut exploits, offtrack behavior at corners and ramp launches, and whether these are the intended final scientific semantics.
- Migrate and preregister the canonical scientific definition/semantic contract only *after* validating candidate behavior.
- Implement verified TRAIN-only pilot executor with 25k evaluation checkpoints, 200k requested steps and fresh calibration/confirmation models and provenance. Legacy WP7 4/4 reports are lost; never represent this implementation as having reproduced them.
- Preserve new artifacts/checkpoints durably outside Git-ignored `artifacts/` and **do not run held-out learned-policy evaluations or final seeds**.
- WP8 remains blocked until a separately reviewed scientific freeze and explicit approval.
