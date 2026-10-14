# WP7/3B research — torque-free 6-DOF kernel, inclined manifold and event budget (2026-10-14)

**Stacked on Draft PR #13. HELD-OUT sealed. No scientific freeze.**

## New research capabilities

- `torque_free_rotation.py`: explicitly registered diagonal body-frame pitch/yaw/roll inertia; RK4 integration of Euler torque-free rigid-body angular equations and unit-quaternion kinematics; fail-closed max spin, finite state, energy and **world angular momentum** drift checks. `BallisticFlightPolicy.torque_free` now enables this in one **narrow, proven high-clearance domain**: no obstacles, matching registered inertia, and a sphere enclosing chassis+wheels remains above the highest road-patch corner for the entire concave-down ballistic timestep. Flight preserves full angular state and produces a per-step audit. Near-road contact or any obstacle is explicitly rejected until a conservative non-SLERP collision sweep is available.
- `rotating_touchdown.py`: inclined 2.5D planar event geometry using signed road-normal distance and normal velocity; gravity projected on the plane normal; conservative handling of road boundaries and initial upward separation after impact.
- `manifold_flight.py`: opt-in full 1–4 wheel impulse on shallow planar inclines already authorized by `BallisticFlightPolicy.max_patch_tilt_rad`. `max_impacts_per_step` (1 by default, 1–8 allowed) authorizes time-ordered bounded **multiple rebound events**. Each event solves a fresh impulse with momentum/kinetic audits, re-verifies swept road and obstacle geometry, and adds a timed `rebound` transition and per-impact audit history. Insufficient event-time progress, event budget, unsafe body/wheel recontact or failed support remain hard gates.
- Existing `BallisticFlightPolicy.manifold` is opt-in, old single-wheel, flat, and fixed-attitude modes remain unchanged.
- World race interaction remains suppressed for all flight/rebound transition ticks; no fake checkpoints or pickups.

## Unresolved science and technical blockers

1. This PR **does not yet couple torque-free angular integration to near-road event/time-of-contact manifold or obstacle sweep**. The high-clearance spherical bound provides a safe domain for torque-free 6-DOF flight, but any close road or obstacle would require versioned non-SLERP swept-body certificate. Feeding RK4 rotation into existing swept OBB logic without that bound could falsely accept tunneling.
2. Manifold contact uses point-wheel references without measured tire radius, tire patch geometry, friction law, compliance or experimental parameters. PGS approximates complementarity and can reject ambiguous scenarios. Sloped contact is limited by the narrow 2.5D atlas and the registered tilt policy.
3. Grounded XZ/yaw remains legacy bicycle/prescribed; a slope+friction impulse producing significant post-impact yaw rate will not be marked grounded. There is no true 6-DOF ground handling.
4. Multiple bounces are finite and bounded by registered budget; zeno-like rapidly repeating impacts, side-entry, unclear close contacts and stacked decks fail closed. Multi-bounce event localization assumes the constant-world-angular-rate ballistic trajectory, not fully coupled torque-free spinning.
5. Green synthetic tests only establish software invariants and exact-HEAD regression, not scientific validity or physical calibration.
6. No training, no OPEN HELD-OUT, no WP8, no historical model rewrite, no merge into main/recovery. Keep Draft until independent scientific acceptance and preregistered evaluation.

## Evidence protocol

Exact HEAD Windows and Ubuntu GitHub Actions focused physics tests, full pytest, owner-decision validator, compileall and pip check. Record exact commit SHA, individual job logs and known limits before requesting any review.
