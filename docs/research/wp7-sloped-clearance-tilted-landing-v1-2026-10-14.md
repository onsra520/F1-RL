# WP7/3B — Sloped release / swept road clearance / low-tilt touchdown (research v1)

**Scope:** stacked on Draft PR #10; no HELD-OUT, no PPO training, no model/evidence rewrite, no WP8, no scientific freeze.

## Prospective geometry and transition contract

1. **Sloped partial-contact and departure, explicitly opted in.** `AutomaticTakeoffPolicy.max_patch_tilt_rad` is zero by default (backward-compatible flat-only); a bounded positive limit up to 0.35 rad accepts height-consistent 2.5D planar ramps. Each wheel samples its individual world-Y gap from the correct plane, with unloaded wheels producing zero force. The four-wheel last-support transition retains deterministic bounded bisection, dynamic state and angular rate. No tire-radius or measured weight-transfer claim.
2. **Swept 3D chassis/road certification.** The shared `certify_swept_chassis_clearance` applies a conservative 3D OBB-vertex half-space clearance against each *bounded* patch polygon whenever an inflated swept XZ footprint overlaps. Translation + shortest-arc rotation bound each subdivided interval; ballistic sagitta adds `g*h²/8`. Insufficient precision, witnessed body penetration or clearance ambiguities invalidate the rollout. Overhang beyond patch polygons is allowed.
3. **Sloped ballistic trajectory.** `BallisticFlightPolicy.max_patch_tilt_rad` is zero by default. Fixed-attitude flight samples a gravity-exact parabola. Plane-crossing times solve normalized signed gap `((p0-plane_origin)·n)/n_y` under relative vertical rate `v·n/n_y`. Bounded polygon membership is required at wheel/body events. Whole-body swept-plane verification blocks unmodeled side entry or penetration, not just start/end tests.
4. **Rotated obstacle OBB safety.** In each chord interval the known world-Y ballistic sagitta expands each local OBB half-axis by `sagitta*abs(axis_world.y)`, a conservative Minkowski containment bound. Collision/unknown from swept OBB invalidates the physics candidate. Existing spin-rate anti-alias guard remains.
5. **Low-tilt stationary touchdown.** On first wheel contact, enforce configured road slope, body-clearance proof, all-four support travel, tilt and relative normal-closing-speed bounds. Accept a *research-only perfectly inelastic* event map: preserve body orientation and horizontal velocity, set vertical and angular rates to zero, and recover local pitch/roll from quaternion for persistent `SuspensionDynamicsState`. The grounded independent suspension solver processes every remaining positive timestep.
6. Default legacy path and previously opt-in flat-only behavior are preserved. Historical models, training tracks, evidence, and scientific protocol are unchanged.

## Explicit blockers

- The low-tilt impact **is not a validated physical impulse**. It intentionally discards vertical/angular kinetic energy; no momentum-conserving contact manifold, wheel tire radius, per-wheel normal-force impulse or measured restitution. No scientific touchdown parameter has been approved.
- Angular velocity during *touchdown* is still rejected: only fixed-in-flight attitude with modest pitch/roll reaches the landing map. Earlier outward-only rotating flight cannot land until 6-DOF impulse / rotating contact resolution is implemented.
- Geometric certificates apply to registered numerical linear+SLERP segments and gravity parabola, not exact underlying spring-force ODE trajectories or all possible mesh geometry.
- Road atlas is height-consistent 2.5D; stacked decks, vertical steps, overhang roofs and arbitrary terrain meshes remain unsupported. Highly banked/steep slopes beyond policy limits fail closed.
- Ground XZ/yaw, spring preload, masses, inertias, penalties, episode rule and tire forces remain uncalibrated / legacy-prescribed. No TRAIN evidence, robustness lower-bound acceptance, independent review or backup/restore receipt established.
- Event detection uses vertex-plane roots and conservative swept-road rejection; unsupported edge/face-first manifold cases can reject valid physics rather than fabricate an accepted collision.

**Test acceptance:** inspect individual Windows + Ubuntu CI logs for exact PR HEAD (focused/full pytest, registered-owner validator, compileall, pip check). Green CI means software regressions passed — **NOT WP7 scientific acceptance**.

**Restrictions:** All PRs remain Draft until independent protocol/scientific review; HELD-OUT SEALED; WP8 NOT STARTED.
