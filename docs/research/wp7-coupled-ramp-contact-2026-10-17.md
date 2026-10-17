# Coupled planar contact and actual TRAIN Ramp proposal

Parent: Draft PR #30, `7094cb2f15af29a6f80ce0dba8c67cdc2b95628b`.
Branch: `research/wp7-coupled-ramp-contact-v1`.

## Implemented domain

`coupled_longitudinal_v1` advances world-Z translation, world-Y heave and
principal-X pitch from one force field. It replaces prescribed longitudinal
motion only when explicitly selected. All four suspension reference contacts
must remain geometrically supported. Steering, yaw, lateral translation, roll,
partial support, takeoff and airborne recovery are outside this new kernel;
unsupported combinations and inconsistent persistent state reject.

This is a planar rigid body with massless suspension reference contacts, not
wheel-spin/contact-patch dynamics or a general spatial vehicle solver. The
mass, inertia, friction and power parameters are provisional engineering
inputs, without measured calibration or scientific admission.

For each contact, let `g = point.y - road_height(point.x,point.z)` and `d=-g`.
The vertical gap gradient is `n/n.y` and its rate is the gradient dotted with
the point velocity `v + omega cross r`. With `P=mg/4`, the contact reaction is

```text
F_normal = (n/n.y) * max(0, P + k*d - c*g_rate)
U_spring = max(0, P + k*d)^2 / (2*k)
normal_damping_loss_rate = (max(0,P+k*d) - vertical_load) * g_rate
```

The spring potential and normal force use the same geometric gap; damping
clipping remains dissipative. Forces act on the mass centre through their
actual lever arms. Each wheel's longitudinal tangent actuator obeys explicit
`abs(F_t) <= mu*N` and positive drive-power `<= P_max/4` bounds. Braking uses
`tanh(tangent_speed/brake_speed_scale)` and signed actuator work. It has no
static-friction/locked-wheel claim; creep on an incline remains possible.
Environment boost is an explicit external COM force with recorded work; COM
drag and pitch damping enter the dissipation ledger. Gravity is applied once
as a world force, without a separate projected-gravity acceleration.

The exact kinetic energy for this restricted configuration is
`m*(vz^2+vy^2)/2 + I_pitch*omega_x^2/2`. The framework follows standard
[multibody force, energy and virtual-work equations](https://underactuated.mit.edu/multibody.html);
the regularized braking law is a declared numerical approximation, consistent
with the distinction between smooth and static-friction models in
[Simscape friction documentation](https://la.mathworks.com/help/simscape/ug/configuring-gravity-friction.html).
These references do not calibrate F1-RL parameters.

## Integration and evidence

RK4 advances all coordinates and rates together. Each accepted substep stores
four-stage contact forces, integrated force/pitch impulses, actuator work,
dissipation and work-balance residual. Both Newton/Euler impulse residuals and
the numerical energy defect must pass before publishing a candidate. Energy
tolerance scales with kinetic/spring energy and changes in gravitational
potential, excluding absolute gravitational potential, so raising the height
origin cannot loosen it.

Failed work balance deterministically bisects the interval. The default depth
20 (maximum configurable 24), together with 4096 accepted leaves per tick,
bounds work. The actual incline seam reproduced a continuous spring gap but
a discontinuous damper force from the changed normal. Depth 8 could not
resolve its quadrature defect; a minimized regression demonstrates rejection
at depth 8 and accepted refined integration at unchanged tolerances. This is
not a global ODE error enclosure, calibrated force proof or exact TOI solver.

Every accepted leaf also passes the existing bounded-patch body/road/wheel
certificate. Collision and race gates cover the numerical substep chords and
SLERP orientations, not a proved continuous-force physical trajectory. Actual
adaptive times are reused by center segments and transition audits.

World, Simulation and Gym expose the coupled audit only for this opt-in mode.
Physics documents use schema 4 and include the contact policy. The default
`None` policy is omitted from legacy documents, preserving prior schema-1/2/3
identity semantics. Two-action historical controls are unchanged; explicitly
versioned three-action braking accepts the coupled grounded mode. PPO
construction still rejects research physics before allocating a trainable model.

`certify_coupled_ramp_execution()` and CLI `--ramp-coupled-proposal` reuse the
fresh exact-root worker, source/runtime/config fingerprints and pinned replay.
The actual TRAIN Ramp proposal uses `mu=.8`, `max_drive_power_w=100000` and
the existing 0.2 rad proposal tilt domain, all explicitly hashed. It records
both registered and executed physics identities and
`engineering_proposal_not_training_registration`. Its trajectory digest binds
the force/work audits; the certificate additionally summarizes their totals
and maximum normalized residual. Controls are zero-steering, not learned PPO.

## Verification scope and remaining mission

Tests cover flat equilibrium, exact frictionless incline motion, initial
Newton/Euler acceleration, Coulomb/power limits, dissipative braking, smooth
timestep convergence, height-origin invariance, seam refinement, unsupported
states, policy/configuration failures, actual Simulation/Gym telemetry, closed
PPO admission and actual Ramp gates/finish/stop with independent source-bound
replay. Gym follows the same tape to episode completion with valid observations.
The direct quaternion guard also rejects a reproduced tiny-yaw projection bug.

Local full regression: **2104 passed, 2 platform skips**. The 35 new cases cover
the kernel, runtime/Gym/admission boundaries and actual Ramp/replay. Compileall,
26-package dependency compatibility and diff checks passed. Two-axis review
has no remaining material findings after the tiny-yaw fix. Exact-HEAD
Windows/Ubuntu evidence is recorded in the stacked Draft PR and external
milestone receipt once verified. No readiness
weights or capability statuses are raised by this proposal. Registered Ramp
integration, all-track corridor-compliant feasibility, coupled steering and
broader lifecycle/perturbation validation remain open. Real measurements,
trusted independent review, Drive restore and scientific sign-off remain
external. Training, HELD-OUT, scientific freeze and WP8 permissions stay false.
