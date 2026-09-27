# Local-Frame Observation Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert deterministic sandbox perception, vehicle state, and an explicit navigation target into a stable normalized observation tuple for future Gymnasium/PPO integration.

**Architecture:** Add one pure-Python `f1rl.learning.observation` module above sandbox. It owns immutable configuration/target DTOs plus a stateless encoder that preserves sensor order, projects velocity/target vectors into the vehicle-local frame, and emits `N + 9` bounded floats without NumPy, Gymnasium, runtime, or policy dependencies.

**Tech Stack:** Python >=3.11, standard-library `dataclasses`/`math`, existing `Vec3`, `TrackWorld`, `VehicleState`, `SensorSuite`, pytest >=8,<10.

**Spec:** `docs/superpowers/specs/2026-09-27-local-frame-observation-encoding-design.md`

## Global Constraints

- Observation size is exactly `len(config.sensor_suite.rangefinders) + 9`.
- Rangefinder features preserve exact `SensorSuite` declaration order; never sort IDs.
- Each rangefinder distance is normalized by that sensor's own `max_distance` and clamped to `[0, 1]` only for boundary noise.
- Velocity and target direction use vehicle-local `+x right`, `+y up`, `+z forward` axes.
- Velocity components divide by explicit `velocity_scale` and clamp independently to `[-1, 1]`.
- Target distance divides by explicit `target_distance_scale` and clamps to `[0, 1]`.
- `target=None` emits zero direction/distance with `target_available=0.0`; exact target position keeps zeros but `target_available=1.0`.
- Absolute world position, yaw/quaternion, angular velocity, hit IDs, rewards, and route-progress state are not encoded.
- Output is a deterministic immutable `tuple[float, ...]`; all values are finite and within declared bounds.
- `learning.observation` may depend on sandbox types and standard library only.
- It must not depend on runtime, API, drivers, Gymnasium, NumPy, PyTorch, Stable-Baselines3/PPO, FastAPI/WebSocket, React, or Three.js.
- Sandbox must not import `f1rl.learning`.
- Existing sandbox, runtime, and perception behavior must remain unchanged.

## Review Focus

- Zero sensors must still produce the nine non-rangefinder features with stable names and size.
- A rangefinder miss must encode exactly `1.0` even when other sensors use different maximum distances.
- Very large signed velocity components must clamp independently without losing lateral/reverse/vertical sign.
- `NavigationTarget` at exactly the vehicle position must remain distinguishable from `None` through `target_available`.
- Equivalent scenes transformed by a consistent yaw rotation must produce approximately equal observations despite floating-point quaternion noise.

---

### Task 1: Observation configuration, target DTO, and feature layout

**Files:**
- Create: `f1rl/learning/__init__.py`
- Create: `f1rl/learning/observation.py`
- Create: `tests/learning/test_observation_contract.py`

**Interfaces:**
- Consumes: existing `SensorSuite` and `Vec3`.
- Produces: `ObservationConfig`, `NavigationTarget`, and `ObservationEncoder.feature_names` / `.size`.
- [ ] **Step 1: Write failing configuration and immutability tests**

Prove `ObservationConfig(sensor_suite, velocity_scale, target_distance_scale)` accepts a real `SensorSuite`, rejects non-`SensorSuite` values, rejects boolean/non-numeric/non-finite/zero/negative scales, and is frozen. Prove `NavigationTarget(position)` preserves the supplied `Vec3` and is frozen.

- [ ] **Step 2: Write failing layout metadata tests**

For sensor IDs intentionally out of lexical order, assert `feature_names` begins with `rangefinder:<id>` in declaration order, followed exactly by the nine fixed names from the spec. Assert `size == len(feature_names) == sensor_count + 9`, including an empty suite.

- [ ] **Step 3: Run RED**

Run: `python -m pytest tests/learning/test_observation_contract.py -q`

Expected: import/collection failure because observation APIs do not exist.

- [ ] **Step 4: Implement DTOs and metadata**

In `f1rl/learning/observation.py`, add frozen/slots dataclasses with exact signatures from the spec. `ObservationEncoder.feature_names` derives only from immutable config and must not sort sensor IDs; `size` returns `len(feature_names)`.

- [ ] **Step 5: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/test_observation_contract.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 6: Commit**

Commit message: `feat(learning): define observation contract`

---
### Task 2: Rangefinder and local-velocity encoding

**Files:**
- Modify: `f1rl/learning/observation.py`
- Create: `tests/learning/test_observation_motion.py`

**Interfaces:**
- Consumes: Task 1 `ObservationConfig`, `ObservationEncoder`, existing `SensorSuite.sense(...)`, `TrackWorld`, and `VehicleState`.
- Produces: encoder internals that generate ordered normalized rangefinder features plus normalized local velocity features.

- [ ] **Step 1: Write failing rangefinder normalization tests**

Configure multiple sensors with different `max_distance` values and prove `encode(...)` uses each corresponding spec position: zero-distance hit → `0.0`, miss → exactly `1.0`, intermediate distances → linear fractions, and declaration order is preserved rather than lexical order or hit order.

- [ ] **Step 2: Write failing local-velocity tests**

Use rotated vehicles with signed world-space velocities and assert the three velocity features equal right/up/forward dot products divided by `velocity_scale`. Include components beyond the scale and prove independent clamping to `[-1.0, 1.0]` with signs preserved.

- [ ] **Step 3: Run RED**

Run: `python -m pytest tests/learning/test_observation_motion.py -q`

Expected: failures because `ObservationEncoder.encode(...)` does not yet emit the rangefinder/velocity contract.

- [ ] **Step 4: Implement rangefinder and velocity prefix**

Add `encode(track: TrackWorld, vehicle: VehicleState, target: NavigationTarget | None) -> tuple[float, ...]` and private standard-library helpers as needed. Obtain readings through `config.sensor_suite.sense(track, vehicle)`, zip them positionally with specs, compute vehicle local basis through quaternion rotation, and append clamped local velocity features.

- [ ] **Step 5: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/test_observation_motion.py -q`
- `python -m pytest tests/learning/test_observation_contract.py -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 6: Commit**

Commit message: `feat(learning): encode sensors and local velocity`

---
### Task 3: Navigation target, grounded flag, and complete encoder bounds

**Files:**
- Modify: `f1rl/learning/observation.py`
- Create: `tests/learning/test_observation_target.py`

**Interfaces:**
- Consumes: Task 2 encoder prefix and Task 1 `NavigationTarget`.
- Produces: complete `N + 9` observation tuple matching `feature_names` exactly.

- [ ] **Step 1: Write failing target-direction tests**

For identity and yaw-rotated vehicles, assert target direction is projected into local right/up/forward coordinates. Include a target exactly at vehicle position and assert direction `(0,0,0)`, distance `0.0`, and `target_available=1.0`.

- [ ] **Step 2: Write failing target-distance/absence tests**

Assert distances below scale divide linearly, distances above scale clamp to `1.0`, and `target=None` emits zero direction/distance with `target_available=0.0`.

- [ ] **Step 3: Write failing grounded and output-bound tests**

Assert grounded is exactly `1.0`/`0.0`; for representative extreme velocity and target inputs, every output is finite and lies in `[-1.0, 1.0]`, with nonnegative sensor/distance/flag features within `[0.0, 1.0]`. Assert `len(encode(...)) == encoder.size`.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/learning/test_observation_target.py -q`

Expected: failures because target/grounded suffix is incomplete.

- [ ] **Step 5: Implement target and grounded suffix**

When target exists, compute `delta`, magnitude, unit direction only when magnitude > 0, project direction into the same local basis, clamp components for floating noise, normalize/clamp distance, and append availability and grounded flags. When target is `None`, append the exact zero/availability contract.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/test_observation_target.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(learning): complete local-frame observation encoding`

---
### Task 4: Observation acceptance, rotation invariance, and dependency boundary

**Files:**
- Create: `tests/integration/test_observation_encoding.py`
- Modify: `f1rl/learning/README.md`

**Interfaces:**
- Consumes: complete Tasks 1-3 observation APIs plus existing sandbox perception/vehicle/track types.
- Produces: a verified observation baseline ready for action mapping and a future Gymnasium adapter.

- [ ] **Step 1: Add complete multi-sensor acceptance flow**

Build a track with multiple obstacles, a non-lexically ordered `SensorSuite`, signed vehicle velocity, and an explicit `NavigationTarget`. Assert the full tuple against the documented feature layout and verify `feature_names` identifies every index.

- [ ] **Step 2: Add determinism and immutability acceptance test**

Encode repeatedly and through two equivalent encoders/configurations; assert exact tuple equality and prove track, vehicle, sensor suite/specs, target, config, and encoder remain equal to their pre-encode values.

- [ ] **Step 3: Add rigid-yaw rotation-invariance test**

Construct two physically equivalent scenes related by the same yaw rotation applied to vehicle orientation, vehicle velocity, obstacles, and navigation target. Assert observation tuples are approximately equal with a tight floating-point tolerance.

- [ ] **Step 4: Add source/dependency boundary checks**

Scan `f1rl/learning/observation.py` for forbidden runtime/API/driver/Gymnasium/NumPy/PyTorch/SB3/PPO/FastAPI/WebSocket/rendering imports/tokens. Scan all `f1rl/sandbox/**/*.py` and prove none import `f1rl.learning` or `..learning` equivalents.

- [ ] **Step 5: Update learning README**

Document the implemented observation encoder, `N + 9` layout, local-frame convention, explicit `NavigationTarget`, normalization scales, and the fact that route-progress selection, action mapping, Gymnasium, rewards, and PPO remain future layers.

- [ ] **Step 6: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- dependency/source scans from Step 4
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden dependencies, and clean whitespace checks.

- [ ] **Step 7: Commit**

Commit message: `test(learning): verify deterministic observation encoding`
