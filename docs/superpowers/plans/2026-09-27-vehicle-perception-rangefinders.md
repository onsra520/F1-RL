# Vehicle Perception & Rangefinder System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic analytic obstacle sensing from vehicle-mounted rangefinders, preserving raw physical measurements for future observation encoding.

**Architecture:** Extend `OrientedBoxVolume` with analytic ray intersection, then build immutable rangefinder definitions/readings and an ordered `SensorSuite` above `TrackWorld.obstacles`. Perception stays inside `f1rl.sandbox`, returns raw values only, and never depends on runtime, learning, API, rendering, or stochastic state.

**Tech Stack:** Python >=3.11, standard-library `dataclasses`/`math`, existing `Vec3`, `Quaternion`, `TrackWorld`, `VehicleState`, pytest >=8,<10.

**Spec:** `docs/superpowers/specs/2026-09-27-vehicle-perception-rangefinders-design.md`

## Global Constraints

- `OrientedBoxVolume.ray_distance(origin, direction, max_distance)` uses analytic slab intersection; no ray sampling/stepping.
- Ray direction must be unit length within absolute tolerance `1e-9`.
- Rangefinder directions are local vehicle-frame vectors; sensing rotates them through `vehicle.orientation`.
- Rangefinders sense only `TrackWorld.obstacles`; all trigger/collectible entities are ignored.
- No-hit returns exactly the configured `max_distance` with `hit_id is None`.
- Sensor declaration order is preserved exactly because it will define future observation-vector layout.
- Equal-distance obstacle hits resolve by lexical `obstacle.id`.
- Perception remains deterministic, immutable, raw, and model-agnostic.
- `f1rl.sandbox.perception` must not import runtime, learning, API, drivers, Gymnasium/PPO, FastAPI/WebSocket, React, or Three.js.
- Existing physics, TrackWorld, world-step, and runtime behavior must remain unchanged.

## Review Focus

- Rays parallel to one or more box axes must still hit correctly when the origin lies inside those slabs, and reject only when outside a parallel slab.
- An origin exactly on an OBB boundary is already inside by the existing inclusive `contains()` contract and must return `0.0`.
- A hit exactly at `max_distance` is included while even a slightly farther hit is excluded.
- A rotated vehicle must rotate sensor direction in world space without modifying either the vehicle or the sensor spec.
- Duplicate sensor IDs must be rejected without sorting or otherwise changing the caller-declared sensor order.

---
### Task 1: Analytic ray intersection for oriented boxes

**Files:**
- Modify: `f1rl/sandbox/track/volume.py`
- Create: `tests/sandbox/track/test_volume_raycast.py`

**Interfaces:**
- Consumes: existing `Vec3`, `Quaternion`, and `OrientedBoxVolume(center, half_extents, orientation)`.
- Produces: `OrientedBoxVolume.ray_distance(origin: Vec3, direction: Vec3, max_distance: float) -> float | None`.

- [ ] **Step 1: Write failing ray-geometry tests**

Add tests for axis-aligned exact distance, rotated-box exact distance, origin inside, origin exactly on boundary, miss, tangential hit, hit exactly at max distance, beyond-range miss, and parallel-axis hit/miss cases.

- [ ] **Step 2: Write failing validation tests**

Prove `direction` rejects zero/non-unit vectors outside absolute tolerance `1e-9`, and `max_distance` rejects negatives, `nan`, `inf`, and booleans while accepting `0.0`.

- [ ] **Step 3: Run RED**

Run: `python -m pytest tests/sandbox/track/test_volume_raycast.py -q`

Expected: failures because `ray_distance()` does not exist.

- [ ] **Step 4: Implement `ray_distance(...)`**

Transform the ray into the OBB local right/up/forward basis using dot products. Use slab entry/exit intervals, handling parallel axes without division; return `0.0` immediately when `contains(origin)` is true. Include hits at exactly `max_distance`.

- [ ] **Step 5: Run GREEN + regression suite**

Run:
- `python -m pytest tests/sandbox/track/test_volume_raycast.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 6: Commit**

Commit message: `feat(perception): add oriented-box ray intersection`

---
### Task 2: Rangefinder specification, reading, and single-sensor sensing

**Files:**
- Create: `f1rl/sandbox/perception/__init__.py`
- Create: `f1rl/sandbox/perception/rangefinder.py`
- Create: `tests/sandbox/perception/test_rangefinder.py`

**Interfaces:**
- Consumes: Task 1 `ray_distance(...)`, `TrackWorld`, `VehicleState`, `Vec3`.
- Produces: immutable `RangefinderSpec(id: str, local_direction: Vec3, max_distance: float)`, immutable `RangefinderReading(sensor_id: str, distance: float, hit_id: str | None)`, and `sense_rangefinder(spec: RangefinderSpec, track: TrackWorld, vehicle: VehicleState) -> RangefinderReading`.

- [ ] **Step 1: Write failing value/validation tests**

Prove `RangefinderSpec` rejects blank IDs, zero/non-unit directions, non-finite/zero/negative ranges, and booleans; valid instances are frozen. Prove `RangefinderReading` preserves raw supplied values and is frozen.

- [ ] **Step 2: Write failing sensing tests**

Add tests proving:
- no hit returns exactly `spec.max_distance` and `hit_id is None`;
- nearest obstacle wins even when a farther obstacle has a lexically earlier ID;
- equal-distance obstacles resolve by lexical ID;
- starting inside an obstacle reports `0.0` with that obstacle ID;
- a forward local sensor rotates with vehicle orientation and hits the corresponding world-space obstacle.

- [ ] **Step 3: Add non-solid exclusion test**

Create overlapping boost, slow, checkpoint, finish, and score-pickup volumes without obstacles and prove `sense_rangefinder(...)` still reports no hit.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/sandbox/perception/test_rangefinder.py -q`

Expected: import/collection failure because rangefinder APIs do not exist.

- [ ] **Step 5: Implement rangefinder APIs**

Validate specs explicitly without normalizing input. In `sense_rangefinder`, use `vehicle.position`, rotate the local direction by `vehicle.orientation`, evaluate only `track.obstacles`, and select the minimum `(distance, obstacle.id)` candidate.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/sandbox/perception/test_rangefinder.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(perception): add vehicle rangefinder sensing`

---
### Task 3: Ordered sensor suite

**Files:**
- Create: `f1rl/sandbox/perception/sensor_suite.py`
- Modify: `f1rl/sandbox/perception/__init__.py`
- Create: `tests/sandbox/perception/test_sensor_suite.py`

**Interfaces:**
- Consumes: Task 2 `RangefinderSpec`, `RangefinderReading`, and `sense_rangefinder(...)`.
- Produces: immutable `SensorSuite(rangefinders: tuple[RangefinderSpec, ...])` with `sense(track: TrackWorld, vehicle: VehicleState) -> tuple[RangefinderReading, ...]`.

- [ ] **Step 1: Write failing construction tests**

Prove `rangefinders` must be a tuple, duplicate IDs are rejected, an empty tuple is valid, and construction preserves the exact caller-declared order without sorting.

- [ ] **Step 2: Write failing sensing-order tests**

Create multiple sensors whose IDs are intentionally out of lexical order and prove `sense()` returns one reading per spec in declared order, with each reading matching direct `sense_rangefinder(...)` behavior.

- [ ] **Step 3: Write repeated-sensing immutability test**

Call `sense()` repeatedly with the same inputs and assert exact tuple equality while vehicle, track, obstacles, suite, and specs remain equal to their pre-sensing values.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/sandbox/perception/test_sensor_suite.py -q`

Expected: import/collection failure because `SensorSuite` does not exist.

- [ ] **Step 5: Implement `SensorSuite`**

Store the exact tuple order supplied after validation. Implement `sense()` as an ordered tuple comprehension over specs using `sense_rangefinder(...)`; do not sort or cache mutable state.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/sandbox/perception/test_sensor_suite.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(perception): add ordered sensor suite`

---
### Task 4: Perception acceptance and dependency verification

**Files:**
- Create: `tests/integration/test_vehicle_perception.py`
- Modify: `f1rl/sandbox/perception/README.md`

**Interfaces:**
- Consumes: complete Task 1-3 perception APIs plus existing `TrackWorld`, obstacle entities, and vehicle orientation/state.
- Produces: a verified raw-sensing baseline ready for `f1rl.learning.observation`.

- [ ] **Step 1: Add multi-sensor vehicle acceptance flow**

Build a track with several obstacles and a rotated vehicle. Use a non-lexically ordered suite containing at least forward/side sensors and prove each returned distance/hit ID matches the expected world-space geometry while tuple order remains declaration order.

- [ ] **Step 2: Add deterministic repeated-run acceptance test**

Run identical sensing repeatedly and across two equivalent `SensorSuite` instances and assert exact reading-tuple equality.

- [ ] **Step 3: Add raw-measurement boundary test**

Prove returned distances remain physical distances rather than normalized fractions and that readings contain no reward, collision-response, runtime, tensor, or rendering metadata.

- [ ] **Step 4: Add dependency/source boundary check**

Scan `f1rl/sandbox/perception/*.py` for forbidden imports/tokens covering `f1rl.learning`, `f1rl.runtime`, `f1rl.api`, `f1rl.drivers`, Gymnasium/PPO, FastAPI/WebSocket, React/Three.js, random, wall-clock, and networking dependencies.

- [ ] **Step 5: Update perception README**

Document the implemented raw rangefinder/sensor-suite contract, local vehicle axes, obstacle-only sensing, declared-order guarantee, and explicit ownership of normalization by `f1rl.learning.observation`.

- [ ] **Step 6: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- dependency/source boundary scan from Step 4
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden dependencies, and clean whitespace checks.

- [ ] **Step 7: Commit**

Commit message: `test(perception): verify deterministic vehicle sensing`
