# Right-turn continuous arc geometry and corridor blocker

Parent: Draft PR #26, `e05b4cc7464da123600520dda0c8f5489f1be98b`.
This increment improves the actual TRAIN geometry search. It does not complete
all-track feasibility or grant research admission.

## Result on the unchanged Right-turn asset

The previous increasing-z graph family could not express repeated passage
through the two authored positive-Z gate planes at z=70. A new arc-length path
can vary heading, pass the checkpoint, return below the plane, and cross the
finish later. It uses the registered 4 x 1.8 x 1.2 m body and steering limit.

All three searched candidates have continuous road-union/body/obstacle and
full-body gate-section certificates. None meets the separate hard 4 m authored
route corridor. The selected candidate has a reproducible violation witness:
center `(x=4.7211648528310235, z=65.2796275735182)`, deviation
`4.720372426481802 m`. Its continuous deviation upper bound is
`4.745302986101824 m`. The certificate records the path coordinate and actual
position of this witness. This disproves compliance of this candidate, not
feasibility of every possible path on the track.

`geometry_status=certified_continuous_planar_path` is explicitly scoped to
body/road/obstacle/gate geometry. `route_corridor.status=hard_violation`,
`executable_feasibility=false` and `training_authorized=false` remain explicit.
The matrix weights/statuses are unchanged; only the blocker description improves.
No canonical track/gate, hard corridor, owner decision or permission changes.

## Path and continuous proof

`ArcPath` stores initial x/z/yaw and bounded `(length, curvature)` segments.
Each segment has unit-speed arc-length derivative `(sin(yaw),cos(yaw))` and
`dyaw/ds=curvature`. Integration uses a stable sinc formula, avoiding subtraction
of nearby trigonometric values. Segments join with continuous position and
heading (C1); curvature is piecewise constant. It is bounded everywhere it is
defined, with finite one-sided limits at joins. No continuous steering-rate or
executed steering-jump claim is made.

For interval length h, every reference position is within h/2 of its midpoint;
every yaw is within `abs(curvature)*h/2`. Every body point therefore moves at
most `(1+body_motion_radius*abs(curvature))*h/2`. Expanding midpoint OBB width
and length by twice that radius (plus numeric guard) encloses the whole moving
body. Existing road-union coverage and OBB SAT verify this enclosure, not isolated
pose samples. Adaptive subdivision improves the sufficient bound. Exhausting
depth or the global check budget yields unresolved.

Directed center roots are analytic: linear projection for lines, trigonometric
plane roots for arcs. Only positive plane-normal velocity is eligible; tangency
is not a crossing. Gates use their actual vertical plane normals and apertures.
Each approximate scalar root also needs a short arc-length bracket with verified
negative/positive center-plane signs. Successive brackets must be disjoint and
ordered; unresolved numerical roots cannot earn passage credit. Brackets are
recorded alongside scalar roots for independent replay.
For each ordered crossing, a connected component of enclosing intervals covers
the complete body/plane passage. The entire body must be negative before it and
positive afterward. Exact polygon-edge/plane intersections of each enclosing
rectangle bound every body section at the gate plane. Merely projecting the
whole body into every same-z gate would incorrectly reject passage elsewhere
on that infinite plane; only the selected complete passage component is tested.

The separate authored-route check uses the existing horizontal-distance API.
Distance-to-route is 1-Lipschitz; midpoint distance plus h/2 bounds each arc
interval. A sampled point strictly beyond 4 m provides an actual violation
witness. Uncertain intervals cannot earn a route PASS. Spawn x/z/heading,
actual world hash, spawn/navigation course hash, body, parameters, path and proof
budget are recorded. Numerical-domain failures reject explicitly.

## Bounded search and API

The deterministic family explores gate approach headings/straight half-lengths
`(23.6 degrees,1.6 m)`, `(24 degrees,1.65 m)`, `(25 degrees,1.65 m)` and uses
curvature no greater than the registered steering limit. Search does not stop
on geometry alone while the corridor fails. It records every candidate's
certificate digest and geometry/corridor result, preferring certified geometry
with the lowest continuous corridor upper bound. Failure of this finite family
is never a global infeasibility certificate.

`audit_train_tracks(..., search_alternatives=True)` dispatches Right-turn to this
family. The default authored-only audit remains unchanged. The nested alternative
schema for Right-turn is now explicitly `wp7_arc_path_certificate_v1`; consumers
must dispatch by schema rather than assume all candidates are x(z) quintics.
Straight/Slalom retain their existing schema; ramp remains unsupported.

Replay the selected `path` with `ArcPath` and `certify_arc_trajectory`. The
underlying certificate digest appears in `search_attempts`; the enclosing search
result has its own digest. No model or policy is loaded.

## Verification and remaining work

Tests cover the real asset and replay, whole-body start/finish passage, narrow
aperture, rotated gate, reversed checkpoint order, tangent touch, thin off-center
obstacle, curvature limits, sloped-domain rejection, malformed inputs, numeric
underflow and the actual four-TRAIN auditor/default compatibility.

General corridor-compliant trajectory search remains required. The existing
candidate violations do not justify changing assets or the 4 m limit. Speed,
braking/slow-zone schedules, executed physics/contact tracking and steering-jump
handling remain unresolved. Ramp support, owner hard-route failure semantics,
empirical validation and all four closed scientific permissions are unchanged.
