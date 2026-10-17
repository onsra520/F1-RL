# Owner-v2 hard-route outcome and reward prefix

Parent: Draft PR #38, `a691ce83fd2090105b62da95a13f69eb60621144`.
Branch: `research/wp7-hard-route-outcome-v2`.

## Owner decision and immutable historical behavior

The explicit owner reply selects strict hard-corridor departure as valid terminal failure, with existing registered reward terms and coefficients unchanged. There is no additional terminal penalty, success credit or reward shaping. Only the eligible event-ordered part of the terminal tick contributes reward. Training and Scientific Acceptance remain blocked.

`configs/research/wp7-hard-route-outcome-v2.json` records these exact choices separately. Normal v2 construction/provenance reads and validates the actual addendum bytes, rejects duplicate/changed fields and any asserted scientific permission, hashes the same parsed bytes and verifies the immutable historical owner-contract SHA. The canonical semantic hash and source-byte hash are both included in race provenance. This owner decision is not a trusted independent scientific approval.

The new versions are `wp7_directed_grounded_owner_v2` and `wp7_directed_lifecycle_owner_v2`, with `hard_violation=terminate_valid_failure`. Older race versions, default registrations and historical C2 behavior remain unchanged; they cannot silently select this rule. Readiness weights and statuses remain unchanged; the old null-decision blocker text is updated to the remaining regression/profile work.

## First-exit geometry and event ordering

Horizontal distance to the authored polyline defines the closed 4m tube. Segment rectangles plus endpoint circles yield a union of allowed path-parameter intervals. The first uncovered interval identifies departure even when a trajectory returns inside before the tick ends. Exactly 4m is inside; outward motion from the boundary has zero eligible duration. An initially outside or numerically unresolved state remains an invalid rollout, rather than a fabricated valid failure.

The new versions require the audited timed trajectory. Ballistic centers have linear XZ projection; accepted contact segments retain the numerical center law. Only gates at or before the first departure boundary are eligible. A valid finish reached while still inside precedes a later route failure and step-limit timeout. Simultaneous ambiguous gate ordering still rejects. Full-tick physical validity/collision guards remain prior to race classification; no unregistered collision response is introduced.

## Reward and observation semantics

The complete physics tick is still integrated and audited. Its accepted end-of-tick state remains the terminal observation and physics telemetry. Race evidence separately records tick-local terminal time, reward-valid-prefix duration/position and full physical tick end. No truncated impulse/work ledger or artificial partial physical state is published.

Existing position-based progress terms evaluate the prefix position; time penalty uses prefix duration; gate/flight events are filtered to that prefix. Registered coefficients and term implementations are unchanged. A zero-duration outward departure has no eligible reward and retains the same component names with zero contributions. Unsupported custom reward terms fail closed because their interior-state semantics have not been established. Untimed pickup/physical events in a shortened prefix reject rather than inventing their event order. Existing C2 offtrack-policy exclusions remain in place.

## Verification and remaining work

The focused suite passes 41 tests. Sixteen new cases include strict 4m and zero-duration boundaries, endpoint caps, leave/reenter, 80 seeded bent-route comparisons to sampled distance, grounded/lifecycle valid-failure integration, finish before/after departure versus timeout, unchanged progress/time/collision coefficients, owner mutation/default-boundary/source-byte rejection and training denial before artifact allocation.

### Spec review

Independent final read-only review reports no material finding within this owner-v2 change; full-tick observation and separate reward prefix are explicit. It did not run tests or assess complete mission acceptance.

### Standards review

Independent final review reports no material finding and independently ran the sixteen new tests successfully. Earlier addendum-read and parse/hash identity findings were corrected with failing-then-passing regressions. Dense-expression/local-import style was noted as a possible maintenance smell, not a correctness defect or a reason for unrelated refactoring.

Full exact-HEAD Windows/Ubuntu regression, hosted CI status and a versioned research execution-profile binding are recorded separately after commit. The feature is opt-in; it does not authorize research PPO training, HELD-OUT evaluation, scientific freeze or WP8, and does not close the complete WP7 objective.
