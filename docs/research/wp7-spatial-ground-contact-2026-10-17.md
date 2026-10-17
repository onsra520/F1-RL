# Supported spatial contact with steering and actual TRAIN Ramp execution

Parent: Draft PR #32, `4c36ce68b0871286293399af21a907a1808d76ed`.
Branch: `research/wp7-spatial-ground-contact-v1`.

## Reproduced registration gap

The actual registered Ramp controller reaches tick 135 at z 50.15428042905613m
before `arc_distance_v1` rejects its inclined contact. This registration selects
the legacy vertical force model, a 0.12 rad tilt domain and horizontal-only
distance bicycle motion. Previous successful Ramp witnesses use separately
hashed engineering proposals. No guard or owner threshold is removed, no
registered profile is silently migrated, and no global infeasibility is claimed.

## Spatial force model

The opt-in `spatial_contact_v1` advances world position/linear velocity,
quaternion and world angular momentum from one force field. With body principal
inertia I and attitude R, omega=R I^-1 R^T H; Hdot is world torque. This retains
full inertia transport instead of projecting away lateral motion, yaw or roll.
Quaternion derivatives use world omega; candidate normalization is included in
the numerical work-defect check. The force/energy formulation follows ordinary
[multibody mechanics and virtual work](https://underactuated.mit.edu/multibody.html).
That reference does not calibrate any project parameter.

Four massless suspension reference points reuse the existing continuous-height
2.5D patch atlas, travel limits and full-body geometry. For gap g=y-h(x,z),
normal spring/damper forces use gradient n/n.y and exact contact-point velocity
v+omega cross r. Spring potential is max(0,mg/4-k*g)^2/(2*k). Gravity is applied
once. COM drag, angular damping and clipped normal damping are dissipative;
environment boost and longitudinal actuation have explicit work ledgers.

Front wheel headings rotate by the steering angle and project into each road
tangent plane. Rear headings stay body-forward. Longitudinal drive/regularized
braking and viscous lateral slip share a Coulomb friction-circle bound; positive
longitudinal power is capped per wheel. These are declared engineering contact
laws, without wheel spin, static-friction, tire-patch or empirical tire claims.

All reference wheels must remain supported within travel. Partial support,
takeoff, flight and landing reject in this new mode. Speed, angular rate, tilt,
principal-inertia triangle inequality and solver budgets are explicit. Existing
lifecycle and longitudinal modes retain their previous domains.

## Numerical audits and runtime wiring

RK4 integrates all state coordinates together. Failed work balance bisects a
leaf at unchanged tolerance and bounded depth/count. Every accepted leaf passes
the bounded-patch body/road/wheel sweep. Audit times generate the same center
segments used by directed race/corridor checks. Numerical sweeps cover chords
and SLERP, not a global physical-ODE error enclosure or exact contact TOI.

PhysicsConfig schema 5 binds the spatial policy and force model. World, Simulation
and Gym publish spatial telemetry only when present. Three-action braking
accepts this explicit mode; research PPO construction remains closed. Absent
spatial policy is omitted from earlier physics documents; the literal legacy
lifecycle hash remains unchanged. No historical C2 action/reward migration.

Runtime validation recomputes linear/angular impulse residuals from accepted
poses and full inertia, checks leaf work arithmetic and recomputes summary
totals. Finite work fields, positive tolerances and nonnegative dissipation are
required. A reproduced infinite-ledger bypass is also fixed in the shared
planar coupled validator. Impulse residuals remain numerical consistency checks,
not independent measurements or proof that the chosen force law is physical.

## Actual execution and verification

`certify_spatial_ramp_execution()` reuses the exact-root fresh worker, runtime/
source/config fingerprints and retained-pin replay. CLI `--ramp-spatial-proposal`
selects `wp7_executed_spatial_ramp_proposal_v1`, including actual and registered
physics hashes plus spatial work audits bound to the trajectory digest. The
proposal retains 0.2 rad tilt, mu 0.8 and 100000 W, adding explicit yaw inertia 1800 kg m²
and lateral damping 12000 N s/m; all remain uncalibrated engineering parameters.
Nominal controls are zero-steering; separate runtime tests exercise steering.

The actual Ramp completes gates, full-body road/contact/aperture checks and
braking/stop, then follows the same tape through Gym completion with valid
observations and exact worker replay. Kernel tests independently cover initial
steering Newton/Euler acceleration, nonplanar roll/pitch/spin Euler acceleration,
planar-kernel equivalence, yaw covariance, mixed-normal lateral impulse,
friction/power/dissipation, height-origin invariance and refinement exhaustion.
Local full regression: **2155 passed, 2 platform skips**; 35 new cases.
Compileall, dependency compatibility (26 packages) and diff checks pass. Two-axis
review has no remaining material Standards or Spec findings.
Negative runtime cases reject mixed modes, missing/forged impulse evidence and
nonfinite/self-declared work summaries. Exact milestone regression/OS evidence
is recorded in the stacked Draft PR and external receipt after verification.

Registration of a coherent all-track profile, randomized full-path evidence,
partial-support/flight integration, measured calibration and independent review
remain open. No matrix weights/statuses, canonical assets, owner choices or
scientific permissions are changed; all four permissions stay false. No new
research PPO training, HELD-OUT rollout, freeze or WP8 execution occurs.
