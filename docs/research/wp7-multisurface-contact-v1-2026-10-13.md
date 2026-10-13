# WP7 multi-surface wheel contacts v1 — research-only

Base: `research/wp7-owner-acceptance-v1` (PR #6). This document covers
only synthetic geometry and the opt-in `step_world(..., multi_surface_policy=...)`
road gate. It is **not** a scientific freeze, suspension dynamics calibration,
or permission to train PPO.

## Contract

* Four explicit wheel reference points independently resolve support patches
  and vertical deviations within provisional bump/droop travel.
* Contact is a vertical height-field projection, not a nearest 3D projected
  patch search; remote/elevated decks cannot silently become wheel support.
* Deterministic tie-breaking uses minimal absolute travel then patch ID.
* A 2.5D atlas allows multiple planar surfaces with continuous elevations at
  contact seams. Overlapping incompatible elevations, disconnected gaps,
  degenerate projections, walls and airborne poses fail closed.
* The *whole* projected 3D OBB footprint must lie in the road union, not just
  the four wheels. Body corners must remain conservatively above every relevant
  road plane.
* A linear translation and shortest-arc quaternion sweep is proven via bounded
  midpoint envelopes. Witnessed invalid poses are OUTSIDE; insufficient
  subdivision resolution is UNCERTAIN, **never accepted as CLEAR**.
* Existing `PlanarRampPolicy` and default full-body road gate are unchanged.
  `multi_surface_policy` is opt-in and mutually exclusive with single-patch policy.

## Explicit limitations / next gates

1. This is a *contact certificate for a prescribed vehicle pose* only.
   The existing `compute_motion` and `integrate` functions still use their
   legacy single `TrackWorld.surface_at` contact. No independent spring/damper
   forces, chassis equilibrium solving or tyre normal-force transfer is claimed.
2. The road atlas disallows multi-height overlapping decks and vertical steps.
   It can conservatively reject otherwise legal bridges or complex geometry.
3. Body/road collision is an over-approximation using supporting infinite planes;
   this is safe but may reject valid seam passages near steep ramp transitions.
4. Provisional `max_bump_m`, `max_droop_m`, seam tolerance and interpolation
   assumptions are engineering values, not independently signed-off science.
5. No takeoff, ballistic flight, touchdown impulse, landing, road episode
   policy, prospective provenance, or full-body canonical experiment yet.
6. Run synthetic focused tests plus full Windows/Ubuntu CI for *the exact HEAD*,
   inspect failures, and retain Draft status until external scientific approval.

No HELD-OUT tracks or model ZIPs were accessed or modified. WP8 stays unopened.
