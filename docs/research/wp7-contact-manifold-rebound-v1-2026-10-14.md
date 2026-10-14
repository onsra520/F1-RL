# WP7/3B: Research manifold and rebound v1 (2026-10-14)

**Stacked research-only Draft PR on PR #12. This is NOT scientific acceptance.**

## Auditable implementation
- Full body-frame *diagonal* inertia: registered pitch/roll inertia inherited from independent suspension; **yaw inertia must be explicitly supplied** in `ContactManifoldPolicy`. No guessed yaw inertia, no mass/inertia calibration claim.
- Fixed-order projected sequential impulses for 1-4 wheel-reference contacts. Unilateral normal impulses and an accumulated friction impulse clamped to a Coulomb circle (`||J_t|| <= mu * J_n`). Normal target speed is `-e*min(v_n_initial,0)`. Each wheel normal complementarity, Coulomb cone, total linear/angular momentum updates, and non-increasing instantaneous rigid-body kinetic energy are checked. Exceeding the iteration budget or any failed invariant raises an explicit error.
- Rotating-first-wheel event localization is inherited from PR #12. Manifold membership is based on a narrow signed-plane wheel gap threshold at the first resolved event; overlapping/multi-surface ambiguity and body intrusion fail closed. The registered `BallisticFlightPolicy.manifold` is opt-in and mutually exclusive with the legacy single-wheel impulse mode.
- Fully supported zero-restitution landings can enter the grounded 3-DOF model only after four-wheel/body certification and a **near-zero yaw-rate gate**, because grounded yaw dynamics remain legacy-prescribed. The post-impact suspension receives heave, pitch and roll rates rather than resetting angular rates.
- Strictly separating rebounds with `restitution>0` remain airborne. A conservative lower bound on wheel vertical separation over the *entire remaining timestep* must be positive; a second contact or unresolved rebound path fails closed. The remainder is consumed using exact gravity-only ballistic motion plus certified body/obstacle sweeps; the event is `rebound`, not `landing`, and never triggers legacy point-volume scores.

## Non-negotiable limitations
- Plane contact uses four *wheel-reference points*, not radius-aware tires; no suspension compliance within the instantaneous impact, no contact patch, no rolling-resistance law, no measured friction.
- The projected Gauss-Seidel method is a deterministic numerical approximation, **not** a proof of solving the global contact complementarity problem; budget exhaustion or energy increase rejects valid but unresolved physical states.
- The current flight integration only accepts event localization against horizontal, height-consistent 2.5D road patches. The standalone manifold solver mathematically accepts arbitrary unit normals but this is not a validated sloped-road event mechanism. Mixed-deck contact and road-edge manifolds are not solved.
- Kinetic audit covers only the instantaneous impulse, not full timestep gravity/spring potential energy; gravity induces energy exchange after the event and cannot be compared with the impact-only energy bound.
- Grounded yaw/inertia transition remains gated; rotating flight uses constant world-angular velocity rather than torque-free Euler equations. Rebound continuation is accepted only when guaranteed monotonic clearance, without second contact; general multiple bounce requires a new event-time integrator.
- Parameters such as mass, moments, tire friction, restitution, gap tolerance, timing, spring damping remain research assumptions, not owner-approved scientific thresholds.
- No TRAIN/HELD-OUT data accessed, no PPO training, no historical model modifications, no WP8, no independent scientific acceptance, no merging main/recovery. Keep every stacked PR Draft.

## Required verification
Run focused numerical contact invariants and synthetic World transition tests plus full pytest on the exact current HEAD in Windows and Ubuntu CI, together with owner preregistration, compileall and pip dependency checks. Green regression does not establish physical calibration.
