# Versioned distance-consistent ground motion

Base: `a14cecfac0183fa75beef37c3bae4a55db6f775c`, Draft PR #27.
Branch: `research/wp7-distance-ground-motion-v1`.

## Reproduced defect and scope

Legacy ground stepping computes yaw rate from the incoming forward speed but
translates with the post-acceleration speed. Full braking can therefore produce
zero distance and nonzero yaw. With registered vehicle parameters, incoming
speed 0.5 m/s, full steering/brake and dt 0.1 s, the legacy kernel rotates by
0.009105041497396505 rad while translating zero metres. This is not a valid
bounded-curvature steering manoeuvre. It must not become a research-policy exploit.

The historical `integrate()` entry point and default C2 numerical law remain
unchanged. Explicit `PhysicsConfig.ground_motion_version="arc_distance_v1"`
selects the new reduced model. Research three-action braking requires this
registration at both factory construction and direct Gym construction, including
legacy body-only research options. Ordinary legacy software fixtures remain
available; no historical model weights are converted.

## Reduced model

The selected ground contact must be horizontal. Incoming horizontal velocity
must be forward, no-slip and within the registered speed cap; unsupported
lateral/reverse/overspeed states are rejected rather than projected away.
If the centre has left a platform while rear wheels remain supported, the
ground plane is resolved from actual supported wheel contacts before force
evaluation. No contacts, differing support planes or nonhorizontal support
are rejected; centre-contact absence never silently selects air dynamics.
The longitudinal acceleration computed at tick start is frozen over the tick.
Integrate `v(t)=clamp(v0+a*t, 0, vmax)` analytically, splitting at stop/cap times.
For constant steering, curvature is `tan(steering*max_angle)/wheelbase` and yaw
change is curvature times integrated distance. The endpoint uses a stable sinc
arc expression; final yaw rate is curvature times final speed. Tick rotations
of pi or more are rejected to preserve the shortest-yaw numerical domain.

Suspension and partial-support substeps sample this same distance/speed/yaw law
at their actual time nodes. A vehicle stopped early stays stopped at subsequent
nodes; an accelerating vehicle is checked against the release movement limit
at each actual substep, not just whole-tick average displacement. Suspension
continues to own heave, pitch, roll and their rates. Exact `yaw_rx_rz_v1` angular
mapping is required. Transition audit identifies `arc_distance_ground_v1` and
rejects zero-distance yaw, sideways/backward chords, excessive chord-derived curvature and inconsistent
final velocity/rates for event-free ground steps.

This is a constrained force-frozen bicycle model. It does not validate tire slip,
physical force measurements or continuous rigid-body dynamics. The published
ground path remains piecewise linear centre translation / SLERP between numerical
substeps. Collision, corridor and gate timing certificates cover that numerical
path; they do not certify the exact curved force-law path between samples.
Partial-release bisection likewise belongs to the numerical trajectory. No global
track infeasibility theorem is inferred from a continuous bicycle model.

## Registration and compatibility

Default legacy physics documents retain schema 1, their original field set and
hash identity. The explicit new model uses schema 2 and longitudinal model
`force_frozen_distance_bicycle_v1`; provenance and world options include the
ground-motion version. Current prepared TRAIN scenarios explicitly select the
new version, so new bound protocols/receipts must be rebuilt. Existing receipts
are not silently migrated. The registered action remains two-dimensional;
three-dimensional research braking is an explicit alternative contract.

Allowed research modes are suspension and bounded lifecycle; dynamic tire-yaw
mode cannot be combined with this no-slip law. Lifecycle v1 still requires a
globally coplanar horizontal world. Suspension-mode draft scenarios can contain
later unsupported patches, but attempting a selected sloped ground contact
raises an error. Ramp feasibility remains unresolved. Airborne gravity and the
existing landing/coasting remainder domain are unchanged; this increment does
not add controlled turning during a landing remainder.

## Evidence and remaining requirements

`tests/sandbox/test_distance_ground_integrator.py` covers stop/cap event distances,
drag-free subdivision equivalence, invalid dt, unsupported slope/velocity and
legacy numerical-law preservation. `tests/integration/test_distance_ground_motion.py`
covers real Gym stop/acceleration, both suspension paths, direct/factory guards,
atomic rejection of oversized substeps, transition audit and registered binding.
Two-axis review found and reproduced the partial-edge force-selection and
chord-direction audit gaps above; dedicated negative regressions cover both.
Versioned braking and lifecycle regressions exercise their existing integration.

Local full regression: **2,023 passed / 2 platform skips**; focused integration,
kernel and execution binding: **112 passed**. Compileall, dependency checks
(26 packages) and diff whitespace checks pass. Two-axis review has no remaining
material findings after the reproduced edge-support and audit-direction fixes.
Exact-HEAD Windows/Ubuntu CI is required; consult the stacked Draft PR checks.
No readiness
weights change. Right-turn corridor compliance, ramp/contact continuity,
all-track executable speed schedules, physical calibration and independent
admission remain unresolved. All four scientific permissions remain false.
