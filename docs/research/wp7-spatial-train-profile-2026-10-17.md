# Coherent spatial TRAIN profile and executable selection

Parent: Draft PR #33, 53dfd9cc7f17544a985e593e60a0e5cd02dbbc71.
Branch: research/wp7-spatial-train-profile-v1.

## Explicit profile, retained defaults

The new `spatial_ground_v1` selection binds a separate version-2 experiment
registration and bound protocol. Every real TRAIN factory shares exactly the
supported spatial force/contact model from PR #33, with its unchanged provisional
parameters and explicit three-action braking contract. The default `pv5_v1`
registration, two-action contract and historical C2 remain unchanged. Old model
weights are not migrated. Unknown profiles reject instead of falling back.

The new registration is draft_not_authorized. It retains all six required
lifecycle domains, three paired seeds, two strata, 60 episodes/cell and exact
one-sided CP95 lower bound >=0.90. Its grounded-only implementation does not
satisfy the declared full-lifecycle research target. Calibration, cohort review,
owner hard-route decision and independent scientific admission remain missing.

## Actual execution, reset and replay

`certify_spatial_profile_execution()` reuses the exact-root fresh worker and
source/runtime-bound control-tape verifier. Profile, paired seed index, vector
rank and optional reset index are explicit. A null reset index selects nominal
spawn. Otherwise the existing feasible TRAIN reset sampler supplies the actual
state and complete draw/rejection metadata; no failed trajectory is redrawn.
Registered nominal and actual initial-world hashes are distinct fields.

All four TRAIN assets complete nominal Simulation, ordered owner-directed-center
gates, full-body road/obstacle/contact checks, hard corridor, braking and stop.
Their control tapes also complete actual Gym with valid observations. A declared
randomized case per track (seed indices 0/1/2/0, rank 2, reset index 7) completes
Simulation and Gym. This finite engineering coverage is not full perturbation
coverage or statistical robustness evidence. Ramp keeps its longitudinal
controller; no parameters or controllers were tuned to those draws.

Spatial work ledgers are hashed into the trajectory. Exact replay reconstructs
profile/reset selection and compares the entire pinned record. Reproduced stale
nominal initial-state hashing under randomized reset is fixed and regression
checked. The auditor's schema 3 separates authored-path findings from actual
spatial execution and rejects mixed source/profile identities across tracks.

## Entry points

```
python scripts/wp7_audit_train_feasibility.py --profile spatial_ground_v1 --execute-controls
python scripts/wp7_certify_planar_execution.py --spatial-profile-track ramp-v1
python scripts/wp7_certify_planar_execution.py --spatial-profile-track right-turn-v1 --seed-index 2 --reset-rank 2 --reset-index 7
python scripts/wp7_training_admission.py --profile spatial_ground_v1 --verify-software
```

Admission exits 2. Selected profile follows execution binding, run manifests,
evaluation admission and denial reports. Software verification includes the new
actual profile execution tests; its operational timeout is 1200 seconds, without
changing physics budgets, statistical thresholds or legacy verification timeout.

## Verification and remaining scope

Before final regression: nominal execution/Gym/replay 14 passed; randomized
Simulation/Gym 4 passed; perturbed identity/replay 1 passed; profile/binding/
manifest/protocol 41 passed. Final exact-HEAD regression and compatibility receipts
are external milestone evidence, not implied by these focused checks.

Two-axis initial review: Standards no material findings; Spec identified the
reproduced reset hash issue, subsequently fixed. Spec follow-up was unavailable
due to agent usage limit; no independent re-review is claimed for that fix or
later CLI/auditor additions.

Partial support, flight, landing and recovery remain unsupported in this spatial
profile. Physical continuous-ODE tracking/enclosure and measured calibration are
not established by numerical work/impulse residuals. Readiness matrix weights and
statuses remain unchanged. All four scientific permissions stay false; no new
research PPO training, HELD-OUT access, asset/archive edit or automatic merge.
