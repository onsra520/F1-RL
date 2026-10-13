# WP7 research — automatic platform-edge release and partial four-wheel support v1

Stacked on PR #9 (ballistic flight), which stacks on PR #8 (3-DOF
independent suspension). **This is not scientific acceptance of 3B.**

## Limited grounded-to-air transition contract

A new explicit opt-in `step_world(..., automatic_takeoff_policy=...)`
requires persistent `FlightState`, `SuspensionDynamicsState`, an
`IndependentSuspensionDynamicsPolicy`, the matching `BallisticFlightPolicy`,
and `VehicleBodyGeometry`. Default/legacy pathways remain unchanged.

- Each wheel samples the actual height-consistent road polygon independently.
  Missing XZ coverage sets that wheel force to **zero**, and only that wheel.
- A wheel projected *inside* the road but beyond bump or droop limits is
  invalid: such a wheel is **not** evidence of leaving a platform.
- Four springs/dampers feed the 3-DOF chassis heave/pitch/roll integrator,
  with zero support force for missing wheels. Numerics: fixed semi-implicit
  Euler under prescribed XZ/yaw from the legacy bicycle integrator.
- The whole 3D chassis is conservatively checked for swept road penetration.
  **Chassis overhang** is legal while only rear/front wheels remain supported.
  Unlike the all-four-support validator, complete body-footprint coverage is
  not required during an edge release.
- When a fixed substep goes from one or more supported wheels to no supported
  wheels, the first fully unsupported pose is localized by deterministic
  bisection of the stated linear+SLERP substep interpolation. Simulation-time
  departure offset and remaining time are reported separately.
- The new `takeoff` transition is automatic **without an upward-impulse
  trigger**, for an edge exit only. It preserves translational and angular
  velocities; it does not reset pitch/roll to fake a safe launch.
- For nonzero angular velocity, the flight kernel advances orientation at
  constant world angular rate **only when it can prove an outward monotonic
  exit from a single flat rectangular platform**, and conservative chassis
  clearance over the entire residual interval. Otherwise it blocks instead
  of inventing a rotated landing.
- Body–obstacle swept collision uses the existing conservative kernel for
  each partial-ground/air trajectory segment. Flight transition frames
  suppress legacy point-volume checkpoint/finish/pickup scoring.
- `FlightState("airborne")` continues into the existing flight kernel on the
  next step with the same opt-in policy. A rotated landing/impulse is not
  registered and remains an explicit unresolved case.

## Hard limits

1. Only flat height-consistent 2.5D road atlas; slope/step takeoff,
   wheel-radius contacts, and mixed-deck landings are not certified.
2. Bisection gives a **time bracket** within
   `release_time_tolerance_s`, not an exact continuous-time physical
   departure instant. Forces are frozen over each integration substep and
   the release pose is interpolated. No claim of exact ODE event integration.
3. Numerical bounds fail closed near road penetration. Narrow patches,
   out-of-order recontacts or geometrically unprovable transitions can be
   rejected rather than incorrectly counted as CLEAR.
4. Moving rigid-body flight assumes constant angular rate, not Euler's full
   rigid-body free-rotation equations; rotating touchdown is deliberately
   unsupported. Landing impulse and touchdown thresholds are not calibrated.
5. Road support exists by wheel reference point (not measured tire contact
   patches or normal-force estimation); `mg/4` reference preload is provisional.
6. A departure via high vertical speed over a road interior is **not**
   inferred by this platform-edge model; only actual wheel XZ support loss.
7. Ground XZ/yaw is still the legacy centre-surface bicycle model. Ordered
   race policy, science trial preregistration and provenance remain pending.
8. No HELD-OUT, no training, no historical model overwrite, no backup-restore
   acceptance, no WP8, no scientific freeze.

Verification gate: focused synthetic tests, full regression, owner validator,
compileall and pip check on the **exact HEAD** for Windows and Ubuntu.
A successful CI run only establishes software regression, not physical fidelity.
