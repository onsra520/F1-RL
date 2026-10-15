# WP7-PV5 Engineering & Scientific Readiness Report

Date: 2026-10-16. Base: PR #19, `24a16f16463a361aec14323e9eb33cc9d48b5390`.
Branch: `research/wp7-unified-physics-admission-v1`. This is an engineering
advance with scientific training admission **BLOCKED**, not a physical
validation, PPO performance or scientific-acceptance claim.

## A. Unified Physics Architecture

`wp7_lifecycle_v1` registers a bounded sequence through Simulation and Gym:

```text
grounded suspension/driving -> partial wheel support -> edge release/takeoff
  -> gravity + constant-world-angular-velocity flight
  -> bracketed reference-wheel/plane event -> audited contact impulse
  -> rebound flight OR representable four-wheel grounded recovery
  -> suspension/driving on the following tick
```

Supported road geometry is horizontal coplanar finite patches. Lifecycle
configuration requires exact quaternion/rate mapping `yaw_rx_rz_v1`, explicit
manifold and automatic-release registration, and near-zero recovery yaw.
Longitudinal and steering/yaw motion remain the declared inherited kinematic
model. This is not a calibrated tire-force or general 6-DOF ground model.

Simulation owns accepted physical state and authoritative tick/time. Suspension
owns heave/pitch/roll; flight owns airborne pose/velocities; the impulse solver
alone changes momentum at impact. Chassis/suspension attitude and rates are
checked, including previously unowned initial yaw momentum. Accepted kinematic
yaw is reconstructed from the accepted heading path. Legacy defaults retain
their original small-angle mapping and behavior.

`info.physics.transition` records mode, initial/final phase, tick start/end,
active solver IDs and ordered tick-local events. A phase change without event
evidence rejects. Automatic release shifts remainder events by departure time.
Unsupported combinations and physical recovery reject before publication.

Full pre/postconditions, tolerances, ownership and extension criteria are in
[transition domains](wp7-pv5-transition-domains-v1.md). Dynamic-yaw/takeoff,
variable-omega impact, sloped/mixed-normal recovery, tire contact patches and
grounded obstacle recovery remain unsupported. The new experiment preparation
uses the existing grounded race/reset domain and explicitly records missing
flight/race/reset integration; it does not disguise it as lifecycle compatibility.

## B. Contact & Recovery Validation

The new opt-in coplanar manifold extension combines finite domains of the same
horizontal plane. It assigns one contact per wheel, resolves shared-patch ties
deterministically by patch ID, and retains patch identity in impulse telemetry.
Old single-patch behavior remains the default. Different-height/normal domains
are rejected. Restitution rebound must certify separation throughout the
remaining tick or use the already bounded, ordered multi-impact mechanism.

Independent symmetric-drop references check:

```text
t = (sqrt(v_down^2 + 2*g*drop) - v_down) / g
J_vertical = mass * (1 + restitution) * incoming_speed
K_after = restitution^2 * K_before
```

The manifold checks total linear/angular impulse residuals, unilateral normal
impulses, Coulomb bounds, convergence and non-increasing kinetic energy within
registered numerical tolerances. First-root monotonicity is bounded or refined
left-to-right; an exhausted bisection budget cannot claim the requested time
accuracy. This strict locator is opt-in at the new lifecycle/coplanar version
boundaries. Event progress uses seconds rather than a velocity tolerance.
Exact suspension angular-rate mapping prevents a finite
pitch/roll landing from being reconstructed through an inconsistent Jacobian.

Clearance, localized contact and unresolved contact remain distinct. An exact
Euler-ODE clearance bound is not a TOI solver. RK4-anchor/SLERP sweep evidence
certifies that defined numerical trajectory. Neither can manufacture impact
from an uncertain interval. Synthetic tests do not establish empirical validity.

## C. Track & Race Feasibility

`python scripts/wp7_audit_train_feasibility.py` reads exactly four TRAIN assets,
binds actual source/world/manifest/config hashes, and performs OBB road-union,
obstacle, gate-aperture, direction/order and junction checks. It does not load a
model, run a rollout, modify an asset or access HELD-OUT geometry.

| TRAIN track | Body-aware finding | Remaining limitation |
|---|---|---|
| straight-sprint | Authored flat segment sweeps and directed full-body gate passage certify | No measured dynamics or registered speed/braking schedule |
| slalom | Individual authored segment body/obstacle sweeps clear | Polyline heading jumps are not bounded-curvature transitions; alternative smoothing not searched |
| right-turn | Near-corner body sweeps leave road union; second checkpoint/finish do not cross their authored local +Z planes | 90-degree authored junction versus ~5.49 m bicycle radius; no alternative-route feasibility claim |
| ramp | Flat road-union certificate does not cover inclined/height-changing atlas; normal transitions identified | Continuous supported ramp dynamics/contact contract required |

Gate passage uses the complete OBB's affine projection along translation onto
the gate plane. This covers oblique incidence; projecting only the body at the
center-crossing pose would falsely accept some narrow gates. Heading-junction
rotation and sloped support remain explicit blockers rather than assumed clear.

[Race-failure decision v1](wp7-pv5-race-failure-decision-v1.md) distinguishes
successful finish, valid truncation, valid terminal failure, invalid physics and
uncertified route. Research accepted steps carry machine-readable outcomes.
Rejected physics/route steps raise, return no PPO sample, retain a defensive-copy
diagnostic and require reset. Physics rejection precedes finish/timeout; accepted
finish precedes step limit. Valid terminal failure for hard route violations is
**not enabled** while the owner decision remains null. No penalty is invented.
Legacy C2 6 m/10-step semantics remain unchanged.

## D. Training Readiness

[Protocol registration](../../configs/research/wp7-pv5-protocol-v1.json) is a
distinct draft. Actual factory construction binds full nested physics settings,
vehicle/simulation parameters, race hashes per course, reset hashes per seed,
observations/actions, rewards, episode limits and the explicitly reused 200k
training-profile inputs. Source/test/config/TRAIN bytes and Git state are bound.
Engineering mass/inertia/spring/damper values remain uncalibrated proposals.

The proposed comparison has four single-track policies and one multitrack policy
per seed replicate, three independent paired seed records, two evaluation strata
and 60 stochastic episodes per policy x TRAIN track x stratum: 15 policies,
120 cells and at least 7,200 episodes. Each cell retains one-sided CP95 lower
bound >=0.90. No pooling or familywise confidence claim is made. Cohort and
multiplicity interpretation require owner/independent review. Final model is
primary; checkpoints are diagnostic rather than selected for best performance.
Invalid episodes abort a cell with preserved evidence, rather than redraw.

Evaluation-cell preparation has distinct nominal/randomized factories and
identity-bound environment/policy seed metadata, without running them. Existing
TRAIN reset rank/index streams and feasible-spawn/no-fallback checks remain.
The current evaluator/runner still needs an admitted full-lifecycle contract
before these registrations can become a scientific experiment.

`capture_runtime_manifest` captures current Git/files, Python/dependency versions,
CPU/GPU/CUDA details and explicit checkpoint ZIP hashes. Its strict executable
schema compares canonical JSON and actual file/runtime evidence, rejecting
missing fields, type substitution, drift and modified artifacts. GPU probing is
an external script, preserving the core's no-direct-PyTorch-import boundary.
This is current/future-run capture infrastructure, never reconstructed historical
training provenance. It does not assert that a training run occurred.

`python scripts/wp7_training_admission.py --verify-software` produces a
machine-readable report and an actual focused-test execution receipt. Exit 2
means blocked. Software verification, physical readiness, registration,
authorization and post-training acceptance are separate fields. The report
evaluates actual track/source/runtime inputs; no declaration-only green can
grant permission. This version deliberately has no trusted approval verifier,
so `training_authorized` cannot become true. A separately reviewed verifier,
trust roots, exact protocol/evidence binding and explicit owner/reviewer
approval are required before adding that transition. The owner preregistration
reader remains unchanged and all four permissions remain false.

Runner and PPO create/learn guards attach a structured denial report. The project
SB3 backend also guards direct construction/learning and rechecks loaded or
reattached environments. Prediction-only loading remains possible; it does not
authorize later learning. Missing/malformed vector metadata fails closed. These
are supported project boundaries, not a sandbox against arbitrary external SB3
code or deliberate monkeypatching.

## E. Verification Evidence

Fresh exact-HEAD test/compile/dependency/CI receipts and the stacked Draft PR link
are recorded in the PR after final verification. This document's intermediate
receipts are not a substitute for that final source binding.

Final local Windows Python 3.12.15 regression after review fixes: **1,858 passed,
2 platform-specific skipped** in 63.91 s. Compileall for `f1rl` and `scripts`,
`uv pip check` (23 compatible packages), owner validation and `git diff --check`
passed. Parent-to-working-tree comparison confirms no recovery configuration,
canonical track asset or artifact changes. The new PV5 focused suite has 50
cases; the admission tool's larger focused receipt additionally includes
existing race/reset regressions. Cross-OS CI must be checked on the final PR
HEAD, independently of these local results.

Review found and regression tests reproduced three defects: unowned yaw being
zeroed, center-pose-only aperture acceptance at oblique incidence, and Python
boolean/numeric equality weakening JSON schema validation. All were corrected.
Full regression also caught a prohibited direct PyTorch import; the hardware
probe was moved outside `f1rl` without weakening the existing boundary tests.
The Spec review agent stopped at its usage limit after reporting its two physics/
geometry findings; no claim of a completed independent scientific review is made.

Canonical recovery/C2 configuration, track assets, owner preregistration and
historical model/evidence archives are unchanged. Owner YAML SHA-256 remains
`d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1`.
No exploratory TRAIN execution, HELD-OUT evaluation, WP8 or merge is authorized.
Existing small synthetic PPO smoke tests remain software fixtures only.

## F. Scientific Acceptance Matrix

| Gate | Implemented / software evidence | Empirical / independent scientific evidence | Status and evidence needed |
|---|---|---|---|
| G1 | Bounded lifecycle, coplanar contact, telemetry/invariant and analytical tests | No empirical validation or independent physical sign-off | BLOCKED: required full-lifecycle race/reset domain, general TOI/recovery where required, measured references |
| G2 | Calibration/evidence integrity schemas, explicit provisional parameters | Measurements and independent comparison absent | BLOCKED: mass/inertia/tire/spring/damper data, uncertainty, preregistered comparison tolerances |
| G3 | Proposed 3-seed cohorts/cells, CP checker, reset and seed preparation | No newly admitted policy or stochastic evaluation samples | BLOCKED: admitted protocol, reviewed cohort/multiplicity, genuine >=60 episodes/cell and CP95 >=.90 |
| G4 | Strict source/config/runtime/model manifests and identity-bound cells | No independent complete-run replay | BLOCKED: actual training-start capture, model/evaluator evidence and independent reproduction |
| G5 | Existing local manifest/hash restore tools | No attested private Drive restore | BLOCKED: authorized private destination and independently verified off-device recovery |
| G6 | Machine readiness report, closed approval boundary, technical code review | No scientific reviewer/owner sign-off | BLOCKED: separately reviewed approval verifier/contract and explicit sign-off after required evidence |

## G. Next Executable Milestone

Independent engineering can next implement compatible airborne race/reset
semantics, solve bounded ramp transitions, construct and certify smooth candidate
routes without editing canonical assets, and define a reviewer-approved signed
approval verifier. These require their own explicit version boundaries and tests.

Actual physical measurements are required for calibration and empirical contact/
vehicle validation. Owner decisions are required for hard-route terminal failure,
cohort/multiplicity interpretation, reference tolerances and trusted approval
identities. Private Drive recovery requires real off-device execution. No amount
of synthetic regression can replace those dependencies or authorize training.
