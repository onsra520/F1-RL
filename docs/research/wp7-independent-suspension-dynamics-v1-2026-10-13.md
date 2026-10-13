# WP7 independent-suspension dynamics v1 (prospective / opt-in)

Stacked on PR #7. **Not a scientifically calibrated vehicle model or
permission to begin PPO training.** This increment addresses the contact-only
limitation by providing an actual, deterministic 3-DOF sprung-chassis solver.

## Numerical contract

The independent four reference wheels use registered 2.5D contact geometry
and per-wheel suspension deflection `d = ground_height - wheel_height`
(positive d denotes bump). For each wheel at a supported substep:

`F_i = max(0, m*g/4 + k*d_i - c*v_vertical_i)`

`v_vertical_i ~= heave_rate - pitch_rate*local_z_i + roll_rate*local_x_i`

`a_y = sum(F_i)/m - g`

`pitch_accel = (sum(-local_z_i * F_i) - angular_damping*pitch_rate)/I_pitch`

`roll_accel = (sum(local_x_i * F_i) - angular_damping*roll_rate)/I_roll`

Heave, pitch and roll rates then positions use fixed-substep semi-implicit
Euler. The restoring preload is `m*g/4` per wheel (all four aligned wheels
form nominal flat-rest equilibrium); numerical/physical coefficients are
**explicit engineering inputs, not owner-approved science parameters**.

The calling legacy integrator supplies **only the XZ displacement and yaw**.
This simplified dynamics contract does not replace centre-surface longitudinal
or lateral forces. The suspension solver owns vertical chassis translation,
pitch and roll and persists them via `WorldState.suspension`; no ad-hoc reset
is allowed mid-rollout. Pitch is local +X (nose-down); roll local +Z (right-up).
Finite bump/droop, deterministic patch ownership, full-body footprint/clearance
and continuous per-substep linear+SLERP sweeps are required. Any unresolved
contact, excessive tilt, unregistered airborne case or failed sweep raises
`SuspensionResolutionError` and cannot masquerade as a valid physics rollout
or obstacle event.

All pose segments are individually inspected against full-body obstacles;
unknown sweep resolution blocks the rollout. A witnessed obstacle hit follows
legacy collision-block behavior, with previous suspension state preserved.

## Hard scientific limitations

- 3-DOF small-angle linearized force / torque, not 6-DOF tire forces or measured
  mass/inertia/spring coefficients; yaw/XZ remains the legacy bicycle model.
- Road atlas requires continuous height-consistent 2.5D surfaces (no stacked
  decks, jumps, discontinuous steps). Four valid wheel support references are
  required for this initial grounded-only model.
- Spring preload `m*g/4` and fixed semi-implicit substeps do not guarantee
  unconditional stability, correct tire wheelbase load transfer, contact
  impulses, or full continuous-time ODE safety.
- Continuous clearance certificate applies to the piecewise interpolated
  **numerical poses**, not exact analytical trajectory between substeps.
- Future work requires physically reviewed tyre radius/contact law, inertia
  calibration, route/ordered checkpoint production integration, hard failure
  semantics and takeoff/flight/landing. No full-body PPO training is permitted.

No HELD-OUT track inputs, model ZIPs, or historical evidence were changed;
scientific freeze remains blocked; WP8 remains unopened.
