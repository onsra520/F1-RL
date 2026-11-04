# Development Auto Drive Collision Terminal — PR #57

Scope: **custom development Auto Drive with Physics v3 only**. This does not
change PPO, TRAIN, HELD-OUT, registered model evaluation or experimental v4
crash/impulse/score contracts.

## Decision

- A geometrically witnessed body/obstacle intersection ends the custom Auto
  Drive episode. No post-impact dynamics or synthetic impulse are simulated.
- The `step_world` legacy failure is preserved for all other consumers. Its
  unmodified `FlightPhysicsError` type, message and failure_code carry extra
  obstacle identities for a deliberately opted-in development runtime adapter.
- The adapter accepts only the witnessed obstacle identifiers from that precise
  failure path. Unknown/uncertain sweeps and other errors still fail closed.
- The terminal frame is **bookkeeping only**. Position/orientation/velocity are
  the **last accepted precontact Physics pose**. The obstacle contact happened
  somewhere within the attempted timestep, not at a certified exact time.
- `reason=collision`, `status=failed`, replay collision IDs/count and
  WebSocket collision event are emitted. The configured `CollisionPenalty` term
  applies once (currently -5.0), with zero fabricated route/time credit.
- `outcome.kind=collision_terminal_development`, `valid_sample=false`,
  `scientific_evidence=false`. The adapter never claims an accepted physical
  impact or a WP7 scored crash. Do not use these data for scientific acceptance.

## Verification

- New cone/barrier regression: requires real Physics witness, failed terminal,
  one collision, unchanged last accepted pose and no scientific validity.
- Unit mapping: configured collision penalty only; no fake impulse or race
  terminal owner event.
- Legacy PPO and v4 remain on their existing exception/certified-crash paths.
- Development clean-lap and longer map performance are separate from the
  collision-terminal correctness contract.

## Limits

No certified first-contact time/position, no impulse, and no post-contact
physical movement. This is intentional for the current finish-or-crash demo.

## Current local results (2026-11-04)

- Cone and barrier full runtime collision regressions, plus direct penalty
  contract: **3/3 PASS**.
- Frontend: **81/81 PASS**, TypeScript typecheck and production build PASS.
- Broader Python set: **46 PASS, 3 FAIL**. All three failures are
  `session.wait(120.0)` timeouts (one v4 jump, two T05 avoidance tracks);
  none reported a wrong terminal, collision count, or physics exception.
  These are **NOT PASS**. Do not infer scientific v4 or T05 acceptance.
- Isolated T05 diagnostic (normal preset): ongoing `running` and steps
  28/45/63/80/98/119 at wall times 10/20/30/40/50/60 s, with no error.
  This shows progress rather than deadlock, but does not explain the
  platform slowdown or certify eventual finish.
- The new PR #57 workflow executes scoped collision and v4 isolation checks
  on a hosted Windows runner. Treat GitHub checks as pending until reported.
