# WP7/3B research — unilateral angular-impact normal impulse v1

**Status: prospective only, opt-in, no scientific acceptance.** Stacked on PR #11. No prior PPO ZIPs, research evidence, training datasets or sealed HELD-OUT materials touched. WP8 remains closed.

## Implemented mechanics

- A *single first-landing wheel* yields an auditable normal impulse derived from (m), registered pitch/roll inertias, contact arm (r), surface normal (n) and restitution (e):
  `v_c = (v + omega cross r) dot n`
  `k = 1/m + (r cross n) dot I_world_inverse(r cross n)`
  `J = -(1+e) v_c/k` when (v_c < 0).
  `v_after = v + (J/m)n`; `omega_after = omega + J I_world_inverse(r cross n)`.
- Mechanical invariants: positive unilateral impulse, normal restitution, no tangential impulse on a flat plane, no increase in instantaneous rigid-body kinetic energy at restitution (e\in[0,1]), deterministic output. An audit records impulse Ns, pre/post contact speed, kinetic energies, and linear/angular deltas.
- Rotating wheel contact localizes a downward first crossing of a finite horizontal 2.5D road patch using fixed segments, interval Lipschitz bounds, recursive subdivision and bisection. Ambiguous wheel boundary-entry, hidden crossings and unresolved geometry reject the rollout; no fabricated impact.
- Swept chassis/road verifier must certify the ballistic motion before contact, plus body–obstacle swept OBB. Single-contact witness must remain unique within a declared gap; otherwise a multi-contact impulse manifold is required. All four spring-travel limits and full-body clearance must be valid before marking grounded.
- Grounded state receives *post-impulse* heave rate and approximated yaw-aligned pitch/roll rates (not forced to zero); preserves orientation and horizontal velocity. Remaining timestep is integrated in the existing independent suspension solver. All relevant scientific coefficients are explicit inputs; old fixed-attitude landing stays unchanged when opt-in not enabled.
- `BallisticFlightPolicy.angular_impulse` is optional, disabled by default. Its angular-speed, bisection, restitution and contact-gap limits are research engineering inputs, **not accepted physical calibration**.

## Fail-closed boundaries

1. **One-wheel normal impact only**. Simultaneous/co-contact, multiple planes, lateral/tangential contact, Coulomb friction, tire radius/normal-contact deformation, interpenetrating body, roadway edge contact, and multi-impulse contact propagation remain unsupported. This is NOT a general rigid-body collision system.
2. **Flat landing patch only**. Sloped angular impact might create yaw torque; yaw inertia and a full rigid-body inertia tensor remain missing from the registered 3-DOF model. Reject rather than assume a value. Fixed-attitude small-tilt sloped landing remains an unrelated provisional event map from PR #11.
3. Constant-world-angular-velocity flight before impact, not torque-free Euler 6-DOF spin integration. Grounded XZ/yaw remains legacy bicycle motion; recovery of rates is small-angle approximation.
4. The wheel *reference* point substitutes for tire contact patch; momentum and energy accounting cover only instantaneous rigid-body impulse, not compliance or subsequent grounded spring integration. Green CI is not proof of scientific mechanical fidelity.
5. Numerical touch event is within a registered time tolerance, not exact analytical contact time for rotating geometry. Ambiguous grazing/corner contacts conservatively rejected. No contact manifold impulse calibration, scientific robustness lower bound, TRAIN rollout acceptance or independent reviewer approval.
6. **No scientific freeze, no PPO training, no HELD-OUT opening, no WP8, and do not merge main/recovery.** Remain Draft.

CI acceptance: exact-current-HEAD Windows and Ubuntu focused/full pytest + preregistration + compileall + pip check. Add solver invariant tests and flight-to-world integration tests; do not treat prior PR CI as current HEAD evidence.
