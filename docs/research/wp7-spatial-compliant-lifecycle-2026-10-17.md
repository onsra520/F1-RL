# Compliant spatial lifecycle above a continuous road atlas

Parent: Draft PR #37, `c785e02356d130ab9f0f8935fc40b04b59ccb53d`.
Branch: `research/wp7-spatial-compliant-lifecycle-v1`.

## Runtime contract

The explicit `spatial_compliant_lifecycle_v1` ground-motion selection extends `wp7_lifecycle_v1` with the existing world-momentum spatial integrator and unilateral v3 law. It is mutually exclusive with the historical ballistic/manifold/release solvers. Physics documents use schema 6 and a distinct solver/model identity; historical documents retain their prior schema and values.

Spring compression determines grounded versus airborne state. Free extension P/k must fit within registered suspension droop. Beyond free extension the spring carries zero force, so chassis/road separation can exceed contact droop without extending a loaded spring past its stop. All four reference projections and the complete body footprint must remain over the continuous 2.5D road atlas. Bump limits and conservative bounded-patch body clearance remain mandatory.

A force-free airborne interval has fixed attitude and exact gravitational translation. Ground boost, rolling drag, steering and tire propulsion do not act in that phase. Nonzero airborne angular rate, uncovered road edges, unproved sweeps and unresolved phase transitions reject. Contact starts continuously at zero spring force/potential; velocity is not reset and no impact impulse/restitution coefficient is invented. The same force/work/impulse ledger covers release, flight, landing and subsequent support.

## Continuous phase evidence

RK stage flags alone cannot prove uninterrupted support or absence of contact. For every ballistic leaf a continuous deflection upper enclosure proves all springs uncompressed. For every unchanged grounded leaf, a lower enclosure proves at least one common compressed-wheel witness over each geometric subinterval; subdivision may change that witness. Missing enclosures force numerical refinement or rejection.

A phase-changing leaf is refined to width at most 1e-8 seconds. Uniqueness additionally requires each wheel's swept projected envelope to encounter only locally horizontal patches at one height. For the adopted linear center / shortest-arc SLERP leaf, each wheel-gap second derivative in normalized time is bounded by theta^2*|r|. Every gap secant must have the same phase-directed sign and magnitude greater than half this bound. Thus all wheel gaps are strictly monotone and aggregate compression crosses once. Unsupported local geometry or ambiguous monotonicity rejects. This is bracketed numerical event timing, not calibrated physical ODE timing accuracy.

Ballistic centers preserve exact gravity for directed gate apertures. Runtime validation binds gravity, initial/final velocity and airborne angular state to accepted poses. Road/body clearance includes gravitational sag enclosure; obstacle checks reuse the existing conservative ballistic obstacle-envelope helper. Unknown/contact intervals reject. Numerical contact leaves retain the existing chord/SLERP sweep interpretation.

## Integrated evidence and limitations

Fifteen new cases cover the complete public-kernel hop/landing/recovery, passive energy and momentum, actual Simulation/Gym phase telemetry, two airborne checkpoints followed by landing and finish, finish before same-tick timeout, DummyVecEnv/SubprocVecEnv reset identity, airborne action handling, unsupported phase/droop/edge/angular states, thin overhead collision, and same-endpoint forged parabola rejection. Hidden ridge and dual-valley regressions reproduce RK sampling gaps and prevent silent missed contact/release.

Focused regression: 76 passed. Two independent read-only review axes found and drove the collision, phase-persistence and trajectory-binding corrections; final specification review reports no remaining material finding within this bounded numerical contract. Full exact-HEAD regression and executable real-TRAIN proposal evidence are recorded externally after commit. The Archify lifecycle map is externally delivered with deterministic validation; browser/perceptual review is pending because its Chrome executable is unavailable.

This does not close rotating flight, road-edge departure, sloped phase onset, mixed-normal landing/recovery, empirical calibration or the complete WP7 mission. Readiness weights/statuses and all scientific permissions remain unchanged. The separately approved hard-route terminal-failure/reward outcome contract is implemented in a following stack; historical C2 and evidence remain immutable.
