# WP7 bounded lifecycle race/reset integration

Parent: Phase 1 Draft PR #23, `81bad5315cc6ba95fb41cef94a050b001055bad1`.
This engineering increment introduces explicit research contracts, without
changing the registered TRAIN experiment or authorizing training/evaluation.

## Registered execution domain

- Physics remains `wp7_lifecycle_v1`: exactly horizontal, coplanar patches,
  constant-world-omega flight, manifold impulses and supported recovery only.
- Race opt-in is `wp7_directed_lifecycle_research_v1`.
- TRAIN reset opt-in is `wp7_train_reset_lifecycle_research_v1`; nominal state
  must be upright, stationary, grounded with persistent grounded flight state.
- Historical race/reset versions keep their former physics-mode restrictions.
  C2 action/observation/reward/episode rules are unchanged.

## Actual pipeline

Manifold flight emits timed exact-gravity center segments, including separate
segments before and after rebound impulses. World stepping composes these with
the actual numerical suspension substeps and partial-support frame times.
Automatic-release flight times are shifted by the localized departure time;
landing appends the grounded remainder at its real tick offset.

Simulation validates full tick coverage, ordered times, spatial continuity and
matching endpoint state before publishing. Snapshots carry this trajectory.
Race consumes it only with the matching lifecycle transition audit. Gravity-only
segments intersect gate planes via quadratic roots, not endpoint chords. Roots
must travel forward and lie inside the actual rotated 3D aperture; tangency and
starting-plane replay cannot earn credit. Roundoff-sized endpoint overshoot can
be clamped only if the actual endpoint has reached the forward plane.

Gate events are ordered by tick-local physical time. Simultaneous ambiguous
gate order is rejected. Gym reports crossing times and their absolute tick-time
origin. XZ corridor checking remains valid on the center chords because the
registered gravity acceleration changes only Y. Body/road/obstacle/contact
certification remains the physics layer's independent responsibility.

The existing 2m soft / 4m hard limits remain fixed. Hard-route violations remain
`uncertified_route_state` invalid samples; no terminal-failure or penalty is
invented. The exact missing owner decision is
`driving.hard_route_violation_semantics` in the preregistered owner file. Any
execution mode requiring an approved terminal-failure rule remains blocked.

## Executable evidence

`tests/integration/test_wp7_lifecycle_race.py` exercises:

- One Gym episode spanning takeoff, airborne checkpoints, audited landing and finish.
- Airborne finish before same-tick timeout, and a thin aperture at exact ballistic height.
- Same-tick automatic edge release followed by an airborne gate.
- Rebound with checkpoint events before/after impact and retained physical times.
- Lifecycle reset replay independent of Gym seed, vector ranks/reset sequence
  through both DummyVecEnv and Windows-spawn SubprocVecEnv.
- Old/new contract mismatch rejection, missing trajectory rejection, unchanged
  hard-route invalidation, and admission denial before artifact creation.

`tests/sandbox/test_timed_center_trajectory.py` independently covers closed-form
ballistic crossings, rotated planes, two-root direction, tangency, invalid
endpoints and floating-point endpoint/no-duplicate behavior.

No ramp/mixed-normal extension or measured physical validation is claimed.
The experiment registration stays grounded-only until its separate review.
Readiness weights are unchanged; only the tested bounded race/reset capabilities
advance. Owner decisions, archives and all four scientific permissions stay closed.
