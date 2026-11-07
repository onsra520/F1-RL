# TRAIN-5 v3 — reset, Physics v4 numerical witnesses and development gates

Source anchor: `8ba236f4ba1f568ad5d6e3f44e9561f47756da74`.
Branch: `handoff/physics-v4-khanh-bang-20261106`.
Status: **new TRAIN-only research preparation; no scientific training authorization**.

## 1. Five-track independent reset implementation

`f1rl/research/wp7_train5_reset_perturbation_v3.py` attaches the already-existing
`TrainResetConfig(version=wp7_train_reset_lifecycle_research_v1)` to all five
frozen source-bound TRAIN-5 factories. It rejects any of the 18 proposed
training seed values for these fixtures. Different rank, reset index, track
and fixture seed produce SHA-256 keyed deterministic independent draw
identities. The original `TRAIN5 v2` file, frozen scenario builder,
checkpoint bytes, 44 locked source objects and HELD-OUT are unchanged.

The fixtures use only seeds 20261107, 20261108 and 20261109. Reset draws
are bounded at +/-0.5 m laterally and +/-5 degrees heading and require
suspension/contact/route feasibility; they fail closed instead of
silently falling back to nominal spawn. Tests verify 5/5 factory support,
reproducibility, non-reuse and reserved-seed denial.

This is a new **engineering fixture protocol**. The 18 seeds remain
available for a *future separately admitted* scientific training protocol.

## 2. Physics v4 numerical validation

`scripts/wp7_train5_v3_numeric_fixture_audit.py` evaluates an immutable
manifest with 30 fixed TRAIN-only reset fixtures (5 tracks x 3 fixture
seeds x 2 reset indices), 12 scripted control steps per fixture. These
are model-free numerical Dynamics tests, **not PPO** and not independent
lab measurements.

All energy/linear/angular impulse residual checks, supported road
contacts, finite action/observation/reward contracts, numerical speed,
source identity and reset provenance are required to PASS; any failure
writes a one-shot FAIL_CLOSED receipt outside the working tree.

The predeclared requirements and development proposal are held at
`configs/research/wp7-train5-v3-fixture-and-development-criteria.json`.
The numerical fixture run is immutable per commit and uses no reserved
training seeds, held-out geometry or frozen PPO evaluation.

**Scientific limitation:** Physics v4 numerical self-consistency is not
measured empirical calibration or proof of reality transfer. The
historical `wp7_calibration_readiness.py` deliberately refuses to
convert SHA-256 metadata into independently measured validation.
Unless source-backed measured evidence becomes available or a distinct,
legitimately narrower simulation-only scientific claim is approved,
global physical validation remains **BLOCKED**.

## 3. Development metrics and scientific admission

Before any development training, the new versioned manifest records:

- A proposed budget of 3,600 episodes over 60 development cells; five
  TRAIN tracks, paired identities, no held-out selection;
- terminal/outcome categories: finish, hard-route violation, certified
  physical crash and invalid/uncertified Physics;
- *proposed* thresholds: nominal finish >=0.90; per-track finish >=0.80;
  hard-route violation <=0.02; certified crash <=0.01;
  zero invalid/uncertified physics; final checkpoint only at 200k steps;
- no threshold is adjusted in response to development results.

These thresholds have **not** been separately owner-reviewed and signed
or validated for sufficient statistical power. They are not an admission
receipt or an implied 6-policy training permit. The existing
`TRAIN5 v2` guard, `TrainingRunner`, and PPO backend still require
independent scientific admission and must not be bypassed.

## Conditions for 6 x 200k PPO training

Independent quantitative physical calibration/validation or explicitly
limited scientific simulation-domain acceptance; a signed frozen
development protocol and evidence-backed threshold review; authenticated
training-specific admission accepted at all three guards; source-commit
CI PASS; and explicit owner approval of the final training-specific
manifest. Current status: `TRAINING_BLOCKED`, 18 reserved seeds untouched.

The registry is closed **with** the previously accepted 15 missing-log
exception; it does not override this separate training scientific gate.
