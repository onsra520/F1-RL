# Track World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic track/world spatial queries for surfaces, solids, trigger zones, and score collectibles on top of the existing 3D physics geometry values.

**Architecture:** Use one renderer-independent oriented-box primitive for solids/zones/pickups, finite planar patches for road geometry, immutable entity definitions, separate collectible runtime state, and a deterministic `TrackWorld` aggregate query. Track-world code reports facts; collision response, race scoring, and RL rewards remain downstream concerns.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`/`math`, existing `Vec3`/`Quaternion`/`SurfaceContact`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-track-world-model-design.md`

## Global Constraints

- World coordinates remain `(x, y, z)` with `+y` as up.
- Track definitions and query results are immutable and deterministic.
- No Three.js, FastAPI, Gymnasium, PPO, NumPy, or third-party collision engine dependency.
- Surface geometry, solid geometry, trigger zones, and collectibles retain separate semantics.
- Score pickup points are game/demo score data, never an RL reward.
- Collision response, lap timing, score aggregation, and trigger-entry event tracking remain out of scope.
## Review Focus

- A point exactly on a patch or volume boundary must be handled consistently and inclusively.
- Equal-distance overlapping surfaces must resolve deterministically by patch ID.
- `max_surface_distance=0` must allow exact-plane contact but reject any separation.
- Duplicate IDs across different categories must fail, not merely duplicates within one tuple.
- A score pickup must emit points once and remain absent from later queries after its ID enters `TrackState`.

---

### Task 1: Shared oriented-box volume

**Files:**
- Create: `f1rl/sandbox/track/__init__.py`
- Create: `f1rl/sandbox/track/README.md`
- Create: `tests/sandbox/track/README.md`
- Create: `f1rl/sandbox/track/volume.py`
- Create: `tests/sandbox/track/test_volume.py`

**Interfaces:**
- Consumes: `Vec3`, `Quaternion`.
- Produces: immutable `OrientedBoxVolume(center, half_extents, orientation)` and `contains(point: Vec3) -> bool`.

- [ ] Write failing tests for identity orientation, translated boxes, rotated boxes, inclusive boundaries, invalid/non-finite half-extents, and immutable construction.
- [ ] Run `python -m pytest tests/sandbox/track/test_volume.py -q`; verify failure because the API does not exist.
- [ ] Implement only the oriented-box behavior defined by the spec.
- [ ] Run the focused test and `python -m pytest -q`; require zero failures.
- [ ] Commit as `feat(track): add oriented spatial volume`.
### Task 2: Finite planar surface patches

**Files:**
- Create: `f1rl/sandbox/track/patch.py`
- Create: `tests/sandbox/track/test_patch.py`

**Interfaces:**
- Consumes: `Vec3`, `SurfaceContact`.
- Produces: immutable `PlanarSurfacePatch(id, origin, forward, normal, width, length)` and `sample(position: Vec3, max_distance: float) -> SurfaceContact | None`.

- [ ] Write failing tests for flat, slope, and ramp samples; outside-width/length rejection; exact-edge inclusion; exact-plane query with `max_distance=0`; distance rejection; invalid axes/dimensions; and deterministic repeated sampling.
- [ ] Run `python -m pytest tests/sandbox/track/test_patch.py -q`; verify failure because the patch API does not exist.
- [ ] Implement projection and finite local-bounds testing exactly as defined by the spec.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(track): add planar surface patches`.

### Task 3: Track entities

**Files:**
- Create: `f1rl/sandbox/entities/obstacle.py`
- Create: `f1rl/sandbox/entities/boost_pad.py`
- Create: `f1rl/sandbox/entities/slow_zone.py`
- Create: `f1rl/sandbox/entities/checkpoint.py`
- Create: `f1rl/sandbox/entities/finish_line.py`
- Create: `f1rl/sandbox/entities/score_pickup.py`
- Create: `tests/sandbox/track/test_entities.py`

**Interfaces:**
- Consumes: `OrientedBoxVolume`.
- Produces: immutable `Obstacle`, `BoostPad`, `SlowZone`, `Checkpoint`, `FinishLine`, and `ScorePickup` definitions with spec validation.

- [ ] Write failing construction/validation tests for IDs, positive boost bonus, slow multiplier `>=1`, non-negative checkpoint order, positive pickup points, and immutability.
- [ ] Run `python -m pytest tests/sandbox/track/test_entities.py -q`; verify failure because entity APIs do not exist.
- [ ] Implement the six entity dataclasses without behavior that mutates vehicles or race state.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(track): add world entities and zones`.
### Task 4: Collectible runtime state

**Files:**
- Create: `f1rl/sandbox/track/state.py`
- Create: `tests/sandbox/track/test_state.py`

**Interfaces:**
- Consumes: `ScorePickup` and later `TrackQuery.score_pickups`.
- Produces: immutable `TrackState(collected_pickup_ids)`, `ScorePickupCollected`, and `collect_score_pickups(pickups, state) -> tuple[TrackState, tuple[ScorePickupCollected, ...]]`.

- [ ] Write failing tests for first collection, exact configured points, multiple simultaneous pickups sorted by ID, repeat collection emitting nothing, input immutability, and duplicate pickup IDs in input failing explicitly.
- [ ] Run `python -m pytest tests/sandbox/track/test_state.py -q`; verify failure because state APIs do not exist.
- [ ] Implement immutable collection state and event generation only; do not aggregate total score here.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(track): add score pickup state`.

### Task 5: Aggregate TrackWorld query

**Files:**
- Create: `f1rl/sandbox/track/query.py`
- Create: `f1rl/sandbox/track/model.py`
- Create: `tests/sandbox/track/test_model.py`

**Interfaces:**
- Consumes: patches, all entities, `TrackState`, `SurfaceContact`.
- Produces: immutable `TrackQuery`, `TrackWorld.surface_at(...)`, and `TrackWorld.query_point(position, max_surface_distance, state) -> TrackQuery`.

- [ ] Write failing tests for nearest-surface selection, lexical-ID tie breaking, gap/no-contact behavior, all entity memberships, collected-pickup filtering, ID-sorted result tuples, duplicate IDs across categories, negative max distance, and repeated-query determinism.
- [ ] Run `python -m pytest tests/sandbox/track/test_model.py -q`; verify failure because aggregate query APIs do not exist.
- [ ] Implement `TrackWorld` as an immutable aggregate; public queries must never mutate definitions or `TrackState`.
- [ ] Run the focused test and full suite; require zero failures.
- [ ] Commit as `feat(track): implement deterministic world queries`.
### Task 6: Track-world acceptance bridge

**Files:**
- Create: `tests/integration/test_track_world_model.py`
- Verify: `f1rl/sandbox/README.md`
- Verify: `f1rl/sandbox/entities/README.md`

**Interfaces:**
- Consumes: complete `TrackWorld` API plus existing physics `compute_motion()` / `integrate()`.
- Produces: verified world-model baseline ready for runtime world stepping and collision rules.

- [ ] Add an integration test containing flat road, an inclined ramp, a gap, and a landing patch; verify the queried contact feeds the existing physics and the gap yields `None`/airborne behavior.
- [ ] Add boost, slow, checkpoint, finish, obstacle, and score-pickup placements; verify each appears only at the intended point membership.
- [ ] Collect the score pickup, assert one `ScorePickupCollected(points=...)`, then query again and prove the pickup is absent while the immutable map definition still contains it.
- [ ] Run `python -m pytest -q`; require zero failures.
- [ ] Run a repeated world-query/collection smoke test and assert exact equality for identical inputs.
- [ ] Scan `f1rl/sandbox/track` and `f1rl/sandbox/entities` for forbidden learning/API/web/renderer imports.
- [ ] Run `git diff --check` and require no whitespace errors.
- [ ] Commit as `test(track): verify track world interactions`.
