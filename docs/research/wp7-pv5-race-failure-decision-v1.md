# WP7 race-failure decision proposal v1

Date: 2026-10-16. Status: **proposed, owner approval absent**.
This record does not edit owner preregistration or authorize training.

## Executed outcome contract

`wp7_episode_outcome_v1` is recorded by research Gym steps. An accepted
sample carries a classification in `info.research_outcome`. Rejected steps
raise and leave `last_research_outcome` inspectable; they return no observation,
reward, termination or success sample. The property returns a defensive copy.

| Outcome | Meaning | Current behavior |
|---|---|---|
| Successful finish | Physics and route accepted, ordered directed gates completed | `terminated=True`, `truncated=False`; precedes timeout |
| Valid truncation | Accepted state reaches registered step limit | `terminated=False`, `truncated=True` |
| Valid terminal failure | Physically accepted action violates a registered race rule | **Not enabled**: owner hard-violation/recovery decision remains null |
| Invalid physical transition | Unsupported, unresolved or inconsistent physics | Raise before race/timeout/reward publication; reset required |
| Uncertified route state | Hard corridor violation, unresolved route sweep or ambiguous gate ordering | Raise, no PPO sample; reset required |

`active` denotes an accepted nonterminal step. `invalid_environment_transition`
separately identifies downstream observation/reward failures: those are not
physics certificates or ordinary race failures either.

## Decision awaiting owner and independent review

Proposed future contract ID: `wp7_hard_route_terminal_failure_v1`.
An independently certified physical trajectory exceeding 4 m from the authored
route would become an ordinary terminal failure, after ordered event comparison
against finish time. An uncertain sweep or unsupported physics would still
invalidate the rollout. Soft >2 m would remain telemetry without an implicit
penalty. No additional reward coefficient is proposed by this implementation.

The reviewer must decide whether the threshold event terminates immediately,
whether any recovery is allowed, how earlier finish and later violation compare,
and which reward composition applies to the final valid transition. Approval
must bind exact physics/race/protocol hashes and the registered event-time
tolerances. This proposal supplies no signature or invented decision values.
Until approved and implemented with tests, the existing invalid-rollout rejection
continues. TRAIN admission remains blocked.

## Geometry scope

`wp7_body_track_audit_v1` audits exactly four registered TRAIN assets using
4.0 x 1.8 x 1.2 m OBB geometry with provisional owner-reference offset +0.3 m.
It checks the complete swept body against road union and obstacles on each
authored linear segment, finite directed gate/body apertures, heading junctions,
surface normal changes and the registered bicycle turning radius. Source bytes,
world identity, manifest and vehicle configuration are hash-bound.

An exact polyline heading jump is not a drivable bounded-curvature trajectory.
The audit does not search alternative curves or claim an entire track is
impossible. Ramp union/support transitions and speed/braking schedules remain
uncertified. Geometric clearance is distinct from dynamic feasibility; brake
remains zero in the policy action contract. No asset is changed to obtain PASS.

Fresh audit results: straight-sprint has a certified authored geometric sweep;
slalom's individual body/obstacle segments clear but heading junctions are not
bounded-curvature transitions; right-turn additionally leaves the road union
near its corner and its second checkpoint/finish do not cross authored local
+Z planes; ramp support/road transitions remain unsupported by the flat sweep.
Every track still lacks a registered speed/braking trajectory and physical
validation. These findings do not approve training.

Focused Windows verification: **108 passed**, covering new audit/outcome tests,
existing physics/race/reset integration, directed crossing, body collision and
road footprint regressions. The full-suite receipt for the preceding physics
increment remains separate; final milestone verification is still required.
