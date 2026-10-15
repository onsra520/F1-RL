# WP7-PV5 transition domains and contact evidence, v1

Date: 2026-10-16. Scope: software research contract, not empirical validation,
training authorization, or scientific acceptance. The owner body remains
4.0 × 1.8 × 1.2 m. This specification defines the bounded lifecycle and the
evidence necessary to extend it; it does not claim a general 6-DOF simulator.

## Source and mathematical basis

The authoritative implementation entrypoints are
[`PhysicsConfig`](../../f1rl/runtime/physics.py),
[`Simulation.step`](../../f1rl/runtime/simulation.py), and
[`step_world`](../../f1rl/sandbox/world.py). The supporting kernels are
[`independent_suspension_dynamics`](../../f1rl/sandbox/independent_suspension_dynamics.py),
[`partial_wheel_release`](../../f1rl/sandbox/partial_wheel_release.py),
[`ballistic_flight`](../../f1rl/sandbox/ballistic_flight.py),
[`manifold_flight`](../../f1rl/sandbox/manifold_flight.py), and
[`contact_manifold`](../../f1rl/sandbox/contact_manifold.py).

The investigation uses the following primary technical sources for the
requirements, rather than treating this implementation as physical ground truth:

- [MIT Underactuated Robotics: spatial momentum and constrained dynamics](https://underactuated.mit.edu/humanoids.html)
  provides the rigid-body momentum framework. For an instantaneous fixed-world
  contact, the required audits are `m(v+ − v−) = ΣJ` and
  `I_world(ω+ − ω−) = Σ(r × J)` at unchanged pose. This application to the
  project's contact solver is an engineering derivation.
- [Drake: dry friction model](https://drake.mit.edu/doxygen_cxx/group__friction__model.html)
  states Coulomb friction with maximum dissipation. The project's simpler
  sequential impulse method is required to check its friction cone and energy
  residuals; satisfying those checks alone does not establish that it solves a
  globally maximum-dissipation contact problem.
- [Drake: contact modeling](https://drake.mit.edu/doxygen_cxx/group__drake__contacts.html)
  explains why rigid frictional contact has numerical and modeling difficulties
  and why compliant alternatives are useful. This motivates retaining explicit
  unresolved cases rather than claiming general contact robustness.

These sources justify audit structure and caution; they do not calibrate F1-RL
mass, inertia, suspension stiffness, friction, or tire geometry.

## State ownership and publication

`WorldState.vehicle` owns the published position, linear velocity, normalized
quaternion orientation, world angular velocity and grounded flag. Persistent
`WorldState.suspension` owns heave, pitch, roll coordinates and their rates while
grounded. During flight it is retained as inactive memory, not advanced as a
second chassis solver. `WorldState.flight.phase` owns the discrete phase.
Wheel contacts and force samples are derived solver outputs; a missing wheel
projection means no support, while excessive travel over an existing projection
is a domain error. Reference points are not measured tire hubs or finite patches.

The suspension model owns vertical/pitch/roll motion only. Legacy integration
prescribes longitudinal and yaw motion for the bounded lifecycle. Optional
dynamic yaw owns lateral/yaw dynamics only in its separately registered grounded
mode and must not be combined with automatic takeoff through a hidden fallback.
Flight owns the ballistic center and its registered orientation law. The contact
solver exclusively owns impact velocity jumps and their impulse/energy audit.

`Simulation` owns integer tick/time. A successful tick advances exactly `dt`;
takeoff/impact times are local offsets in `[0, dt]` and are ordered before
remaining-time integration. The candidate must pass physics and race validation
before state, timestamp, reward, progress, or a finish is published. An exception
must leave the prior simulation snapshot intact. Event uncertainty is an invalid
transition, not a terminal driving failure and not a training sample.

At a transition without impulse, all components of `p`, `v`, `q`, and `ω` must
be continuous within registered state tolerances. At impulse, `p` and `q` remain
continuous; changes in `v` and `ω` must equal the recorded impulse. Grounded
publication additionally requires chassis/suspension height, attitude and rates
to agree under the selected, versioned suspension representation. A nonzero
unrepresentable post-impact yaw component must reject, not be zeroed.

The inherited small-angle suspension mapping uses
`q = q_yaw Rx(pitch) Rz(roll)` but approximates angular velocity using
`q_yaw.rotate((pitch_rate, 0, roll_rate))`. It is not the exact finite-angle
Euler-rate Jacobian. Therefore a new lifecycle must either use that same mapping
consistently and check reconstruction residuals, or introduce a separately
versioned exact mapping. Mixing an exact inverse with the approximate forward
map can change angular velocity at the next grounded tick. A finite-angle
reconstruction failure must be explicit; small tilt alone is insufficient proof.

## Supported transition table

All tolerances below are engineering numerical tolerances, not owner-approved
physical measurement tolerances. The active configuration and its SHA-256 must
identify the actual values. A supported subset requires integration evidence
through Simulation and Gym as well as kernel tests.

| Transition | Preconditions and physical domain | Solver owner and invariants | Postcondition and tolerances | Failure and required evidence |
|---|---|---|---|---|
| Grounded → grounded compression/rebound | Persistent compatible suspension; supported finite 2.5D atlas; all required wheel references within bump/droop; small-angle tilt below policy bound; no body penetration | Prescribed XZ/yaw candidate plus independent spring/damper substeps; unilateral wheel load `max(0, preload+k*d−c*v_wheel)`; finite state | Grounded phase; height/attitude/rates synchronized; force samples and active solver retained; `integration_substeps` and `max_abs_tilt_rad` registered | Reject missing travel/support, unresolved swept body clearance, tilt overflow; equilibrium, heave/pitch/roll response, timestep refinement and state-continuity fixtures |
| Grounded → partial support | Automatic policy explicitly registered; acceptable road tilt; movement/substep ≤ `max_substep_xz_m`; ≥1 wheel remains supported | Partial-support solver owns heave/pitch/roll; unloaded references receive zero force; support may disappear monotonically | Grounded while any support remains; four-slot support mask and loads; no phantom support | Reject newly appearing wheel support, excess bump/droop, atlas ambiguity, chassis contact; one/two/three-wheel masks, edge order and adversarial recontact tests |
| Partial support → takeoff | All wheel projections disappear on a bracketed edge-exit; swept chassis clear; no inferred takeoff merely because travel is exhausted | Partial-support solver localizes release on its prescribed interpolated discrete trajectory, then passes full chassis state to flight | Airborne flag and phase agree; no impulse at release; local departure time and residual `dt` preserved; release bracket ≤ policy tolerance or reject | This is a discrete-model edge-time bracket, not an error bound against the force ODE. Reject unresolved bracket, recontact and unsupported same-tick landing; check `p/v/q/ω` continuity and ordered event timestamps |
| Explicit takeoff → airborne | Registered launch phase, prior certified support, positive upward velocity, launch rotation within the explicit-launch domain | `authorize_takeoff`; no synthetic impulse is invented | Same pose/velocity at launch; takeoff event at local zero then ballistic remainder | Reject rotating explicit launch outside supported path; independent launch-height/time oracle and negative launch tests |
| Airborne → airborne, fixed-world-ω | Gravity-only center, registered constant-world-ω orientation law, finite upward 2.5D atlas, certified obstacle/body/wheel clearance | Ballistic kernel owns `p(t)=p0+v0*t−g*t²/2`, `v(t)=v0−g*t`; orientation follows registered spin; no ground boost/drag | Whole tick consumed, airborne phase; active solver/clearance domain explicit | Reject witnessed contact without eligible solver and unresolved clearance. Closed-form fall and constant-axis spin tests; endpoints alone are insufficient |
| Airborne → airborne, variable-ω torque-free | Positive matching three-axis inertia, no contact over the full interval, angular policy and required clearance policy explicitly registered | Euler/RK4 angular kernel plus numerical sweep; optional exact-ODE enclosure separately certifies separation | Energy/world-angular-momentum drift within registered limits; no landing event earned from clearance | Reject near contact, angular bounds or drift overflow; principal-axis exact cases and precession invariants; this branch cannot authorize variable-ω impact |
| Airborne → localized road contact | Constant-world-ω flight with bracketed first approaching reference-point event; chassis clear up to event; contact speed/tilt within policy; deterministic simultaneous set | Wheel TOI localization identifies event before manifold impulses; normals from actual registered planar patches | Pose continuous; event bracket and wheel IDs retained; `touchdown_time_epsilon_s` applies to time, `simultaneous_gap_m` to geometry | Reject tangency/ambiguous root, chassis-first contact, penetration, missing first contact, mixed-normal or unresolved contact; analytic vertical drop, seam ties and time-order tests |
| Contact → grounded landing | Explicit manifold; nonnegative normal impulses; supported contact geometry; post-contact four-wheel support; post-ω representable by grounded state | Sequential impulse solver owns `v/ω` jump, applies Coulomb bound and momentum/energy audits; no pose snap | Grounded flag/phase; suspension reconstructed from post-impact state with residual checks; remaining tick integrated without losing time; audit history preserved | Reject nonconvergence, energy gain, momentum residual, unsupported yaw/attitude/rates, or support loss; symmetric/asymmetric analytical impulses and nonzero-rate continuity tests |
| Contact → rebound | Restitution >0, auditable manifold; bounded count of impacts; each subsequent event separately localizable or every wheel certified separating for remainder | Impulse solver then ballistic continuation; impact count and local event times monotonic | Airborne after rebound; impulse history ordered; no fake grounded interval; no unconsumed residual time | Reject exhausted impact budget, unresolved subsequent contact or insufficient separation; restitution energy, repeated-event ordering and strict no-tunneling fixtures |
| Grounded recovery → subsequent driving | Landing completed, full state representable and validated; supported road remains; no obstacle recovery required | Suspension regains sole vertical/angular ownership; inherited longitudinal/yaw model remains explicitly prescribed | Next tick starts from landed state; no rate reset or stale suspension reuse | Reject state mismatch before publication; two-or-more-tick landing-to-driving tests with nonzero rates and timestep boundary cases |

The opt-in `ContactManifoldPolicy.coplanar_flat_contacts=True` covers multiple **exactly coplanar
horizontal** authored patches with the same normal and plane height. This merely
joins finite domains of the same plane; it does not solve mixed-normal impacts.
It deduplicates wheels deterministically at a shared seam, retains patch IDs,
reject gaps/steps/tilted normals, and require whole-body/four-wheel support for
grounded recovery. The old single-patch policy must remain version-identifiable.

`wp7_lifecycle_v1` requires the exact `yaw_rx_rz_v1` angular-rate mapping,
constant-world-omega flight, explicit manifold and automatic-release policies,
exactly horizontal equal-height road patches, and recovery yaw residual at most
`1e-8 rad/s`. It rejects torque-free impact and dynamic-yaw combinations. The
exact rate mapping corrects quaternion/rate ownership; spring/damper forces
remain the declared small-angle approximation. It does not calibrate that model.
Legacy policy defaults retain `small_angle_v1` and single-patch contact.

Actual Simulation snapshots and Gym `info.physics.transition` carry
`wp7_transition_audit_v1`, mode, initial/final phase, authoritative tick start/end
time, actual solver IDs and tick-local transition events. A phase change without
its ordered event is rejected. Landing requires four audited simultaneous
contacts, dissipative energy and representable suspension state. Automatic
release shifts any subsequent flight-event times by the departure time.

The inherited non-manifold low-energy landing map deliberately zeros vertical
and angular rates. Its own source states that it is not an impulse/momentum
model. It may remain available under its historical research boundary; it cannot
serve as the momentum-audited unified lifecycle landing path.

## Clearance, contact and uncertainty are different outcomes

1. **Exact-model clearance.**
   [`euler_ode_clearance`](../../f1rl/sandbox/euler_ode_clearance.py) bounds exact
   torque-free angular speed using `sqrt(2E/min(I))` and encloses each material
   point by a radius-scaled angular-motion bound around its initial orientation,
   together with the ballistic center parabola. A positive separating bound for
   the whole interval certifies clearance under those assumptions. It does not
   certify RK4 accuracy, a collision time, tire contact, or impact response.
2. **Defined-numerical-path clearance.**
   [`torque_free_swept_ccd`](../../f1rl/sandbox/torque_free_swept_ccd.py) certifies
   the path made from RK4 angular anchors and shortest-arc SLERP, with ballistic
   chord sagitta accounted for. Its result applies to that prescribed trajectory.
   Small energy/momentum drift is useful but is not a validated global state-error
   enclosure against Euler's exact ODE.
3. **Localized contact in a bounded model.** A reference wheel-to-plane gap must
   have a justified first-event bracket under the registered sampler and motion
   bounds. The bracket width and approach direction support the event; the
   corresponding manifold solver may then apply an impulse. Geometric contact
   tolerance alone does not prove an earlier event was absent.
4. **Unresolved contact.** An enclosure touching an obstacle/road, exhausted
   subdivision, tangential/ambiguous roots, incompatible simultaneous normals,
   or unsupported recovery cannot become a fictitious hit or successful finish.
   Reject the complete candidate and preserve the preceding published state.

For instantaneous impulses at fixed pose require:

```text
J_n >= 0
||J_t|| <= mu J_n
m(v_after - v_before) = sum(J)
I_world(omega_after - omega_before) = sum(r cross J)
K_after <= K_before + energy_tolerance_j
normal velocity/restitution residual <= velocity_tolerance_mps
```

The default manifold tolerances are `1e-5 m/s` velocity, `1e-5 J` energy and
`1e-4 m` simultaneous gap; defaults are not universal validation thresholds.
Impact timing must use seconds, never compare event time with a velocity
tolerance. The complete configuration must register mass/inertias, friction,
restitution, sweeps, maximum post-spin, yaw representability and impact budget.

## Explicitly unsupported domains and measurable exit criteria

| Domain still blocked | Evidence required before extending support |
|---|---|
| Variable-ω true Euler-ODE TOI/contact | Validated trajectory enclosure tied to exact initial state; lower/upper first-impact-time bracket meeting registered seconds tolerance; no missed earlier contact; interval refinement convergence; independent analytic principal-axis oracles plus adversarial asymmetric-precession cases; matching impulse/recovery audits |
| Mixed-normal/mixed-height simultaneous impacts | A deterministic manifold covering every eligible contact/normal, explicit simultaneous-event tolerance and complementarity objective; residual, momentum and dissipation checks; seam/step/corner order independence tests and physically registered domain |
| Finite tire contact patch and unsprung wheel dynamics | Measured hub/radius/width and uncertainty, wheel spin/inertia, rolling/sliding/contact law and support geometry; static load, braking/traction, slip and impact reference measurements; integration tests proving this geometry actually owns contact |
| Grounded obstacle collision/recovery | Certified swept body TOI, actual contact normal/manifold, registered friction/restitution and recovery; no legacy rollback velocity-zeroing; momentum/energy audit and subsequent valid driving tests |
| Dynamic-yaw ↔ automatic takeoff lifecycle | A single compatible ownership law for longitudinal/lateral/yaw and suspension; low-speed transition law; takeoff/landing angular momentum mapping; nonzero-yaw landing/flight tests without overwritten spin |
| General airborne race acceptance | Registered 3D road/corridor/checkpoint semantics and time ordering; uncertainty/failure classification; explicit owner decision, with no reuse of grounded-only success semantics |

Synthetic test completion can establish only the implemented contract. Empirical
readiness additionally needs independently reviewed measured mass/inertia,
suspension and contact references with declared uncertainty and preregistered
acceptance tolerances. Training, HELD-OUT, scientific freeze and WP8 permissions
remain false until their separate approval contracts are verified.

## Evidence inventory and verification responsibilities

Existing direct evidence locations include
[`test_independent_suspension_dynamics.py`](../../tests/sandbox/test_independent_suspension_dynamics.py),
[`test_partial_wheel_release.py`](../../tests/sandbox/test_partial_wheel_release.py),
[`test_ballistic_flight.py`](../../tests/sandbox/test_ballistic_flight.py),
[`test_contact_manifold.py`](../../tests/sandbox/test_contact_manifold.py),
[`test_manifold_flight.py`](../../tests/sandbox/test_manifold_flight.py),
[`test_euler_ode_clearance.py`](../../tests/sandbox/test_euler_ode_clearance.py), and
[`test_torque_free_swept_ccd.py`](../../tests/sandbox/test_torque_free_swept_ccd.py).
These are source references, not a claim that a fresh run passed.

Fresh Windows verification of this implementation increment (2026-10-16):
`python -B -m pytest -q` returned **1,825 passed, 2 platform-specific skips**.
The new lifecycle and coplanar-contact fixtures contribute 17 passing cases.
Owner preregistration SHA-256 remains
`d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1`;
its four admission permissions remain false. This is an intermediate software
receipt, not milestone completion or cross-platform CI evidence.

PV5 integration evidence must additionally demonstrate through Simulation/Gym:
compatible version registration; positive bounded lifecycle execution;
nonzero-rate state continuity; exact timestamp consumption; active solver names;
audit propagation; coplanar seam boundary behavior if enabled; malformed-state,
near-contact, mixed-normal and incompatible-policy rejection; unchanged previous
snapshot on rejection; no reward/progress publication on invalid physics; and
unchanged legacy behavior. The milestone report must record fresh test commands,
results and exact commit, independently of this specification.
