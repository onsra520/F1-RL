# WP7 2B: prospective whole-body road-union footprint guard

**Status:** research implementation, **not an accepted full-body road/route physics specification**.

Parent baseline: `research/wp7-body-collision-v2` (Draft PR #3); new review branch: `research/wp7-road-footprint-v1`.

## Implemented scope

- The **entire XZ orthographic projection of all 8 corners** of the 3D rigid body is covered by the **union of surface polygons**, not merely vehicle centre or four chosen wheel/body corners.
- Convex road-rectangle clipping computes the uncovered footprint area across multiple patches. Adjacent strips may jointly cover a body even when neither does alone; a small gap may not.
- The continuous timestep check assumes the same **linear translation + shortest-arc SLERP rotation** as the collision kernel. An inflated midpoint footprint conservatively encloses all intervening projected 3D body points. A recursive interval may be declared CLEAR only when the envelope is fully inside the road union. A detected crossing is OUTSIDE; an unproved interval is UNCERTAIN and invalidates the rollout rather than being counted as success.
- Accepted cases require **coplanar, upward-facing, horizontal** road patches, and vehicle `grounded=True` at both ends with reference height equal to the patch elevation plus the configured `VehicleParameters.ride_height`.
- Unsupported situations (ramp/slope surfaces, multi-height overlapping road decks, flight, elevation mismatch) return UNSUPPORTED and invalidate the prospective full-body transition. This is **not** a substitute for a future physical ramp and airborne policy.
- The new check is invoked only when `body_geometry` is provided to `step_world`. Previous 200k PPO runs/legacy simulator never provide it. Obstacle response is resolved first, and the whole-body road sweep is validated against the accepted motion; obstacle rejection that stops motion cannot create a spurious off-road crossing.
- Numerical polygon-area tolerance `1e-9 m²` is for floating-point robustness **only**, not a permissive physical corner-cutting tolerance.
- No silent environmental seed randomization, reward or action-vector changes. HELD-OUT remains sealed.

## Precision of scientific claims

**Proved within stated assumptions:** ground-based whole-body XZ projection remains inside the union of horizontal flat surface footprints over the interpolated timestep when returned `clear`.

**Not proved:** all four physical tires maintain contact; suspension, vertical clearance, ramp or airborne support, road friction, corner radii, acceptance of projection over a bridge/tunnel, lateral distance to the authored route, or real-world continuous rigid-body dynamics beyond the assumed interpolation.

The guard presently *raises* for outside/unsupported/uncertain body motion. It is a fail-closed **research validity gate**, **not** an authored road-off penalty / policy-learning termination. Thus full-body policy training must remain blocked. The ultimate semantics of off-road termination versus recoverable failure must be registered prospectively before training; replacing these exceptions with a new truncation/reward policy is a separate work item.

## Target acceptance criteria still open (2B)

1. Specify approved body geometry (dimensions, local centre, tire contact footprint/reference pose).
2. Specify versioned ramp and airborne contact/support model, road topology and allowed traversal of sloped surfaces.
3. Specify **exact** allowable distance from authored route and full-body road clearance, independent of earlier PPO performance. The legacy `6m for 10 steps` centre-route truncation is not the 2B full-body acceptance rule.
4. Model event-volume entry for checkpoint/finish/obstacle/boost where discrete sampling can miss thin triggers.
5. Ensure all 4 TRAIN tracks, including Ramp, can execute under the new rules; otherwise reject the candidate.
6. Bind all simulator/road/route semantics and track hashes to new **prospective** run provenance and immutable experiment profile. Do **not** rewrite historic ZIP models, freeze them under new physics, use HELD-OUT for debugging or whitelist changed training code as retroactive audit-only source.

## Test targets

`tests/sandbox/test_road_footprint.py` covers body overhang with centre still inside, adjacent road union and a gap, 180-degree rotation with a mid-timestep excursion, 3D roll enlarging projected footprint, conservative boundary uncertainty, unsupported ramp/airborne/deck/elevation, invalid numeric contracts and world-step legacy compatibility.

The companion CI workflow runs both body collision and road footprint tests, then the full regression on Python 3.12 Windows + Ubuntu. **Pass/fail must be read from the HEAD commit's actual CI run**, not from the original PR #3 checks.

**Scientific freeze, full-body training, HELD-OUT, WP8: BLOCKED.**
