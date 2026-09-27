# Sandbox Physics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic simplified 3D ground/air vehicle physics for F1-RL, including ramps, airtime under gravity, and non-bouncing landings.

**Architecture:** Keep renderer-independent geometry in `sandbox/geometry`; keep vehicle and surface-contact domain values at sandbox level; split grounded and airborne dynamics; route both through `vehicle_dynamics.py`; integrate all motion through one deterministic integrator. Track/world code will later produce `SurfaceContact` without changing the physics interfaces.

**Tech Stack:** Python 3.11+, standard-library `dataclasses` and `math`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-sandbox-physics-core-design.md`

## Global Constraints

- World coordinates are `(x, y, z)` with `+y` as up.
- Position, linear velocity, and angular velocity are 3D values.
- Orientation source of truth is a normalized quaternion.
- Controls are normalized: steering `[-1, 1]`, throttle/brake `[0, 1]`.
- Ground and air physics are separate and deterministic.
- Ramp launch velocity must emerge from surface-tangent motion; no jump impulse hack.
- Airborne motion uses gravity plus simple linear drag.
- Landing removes inward normal velocity and does not bounce in V1.
- No third-party physics/math dependency is introduced in this milestone.
## Review Focus

- Zero-length vectors and invalid quaternions must fail explicitly instead of producing NaNs.
- Surface normals must be normalized and outward-facing according to the supplied contact contract.
- Ground projection must not silently erase legitimate ramp-generated vertical velocity before takeoff.
- Airborne controls must not alter trajectory in V1.
- Ground/landing projection must preserve tangential position and velocity; it may correct only the surface-normal component.
- Landing must remove only the velocity component directed into the surface, preserving tangential motion.

---

### Task 1: Geometry primitives

**Files:**
- Create: `f1rl/__init__.py`
- Create: `f1rl/sandbox/__init__.py`
- Create: `f1rl/sandbox/geometry/__init__.py`
- Create: `f1rl/sandbox/geometry/vector.py`
- Create: `f1rl/sandbox/geometry/rotation.py`
- Create: `tests/sandbox/test_geometry.py`

**Interfaces:**
- Consumes: none.
- Produces: immutable `Vec3` and `Quaternion` domain values with the operations defined by the spec.

- [ ] **Step 1:** Write failing tests for vector arithmetic, dot/cross, magnitude, normalization, projection/rejection, quaternion identity, normalization, multiplication, axis-angle construction, and vector rotation.
- [ ] **Step 2:** Run `python -m pytest tests/sandbox/test_geometry.py -q` and verify failure because geometry APIs do not exist.
- [ ] **Step 3:** Implement only the geometry operations required by the tests/spec.
- [ ] **Step 4:** Run the focused test and `python -m pytest -q`; require zero failures.
- [ ] **Step 5:** Commit as `feat(sandbox): add 3d geometry primitives`.
### Task 2: Vehicle and surface domain values

**Files:**
- Create: `f1rl/sandbox/vehicle.py`
- Create: `f1rl/sandbox/surface.py`
- Create: `tests/sandbox/test_vehicle.py`
- Create: `tests/sandbox/test_surface.py`

**Interfaces:**
- Consumes: `Vec3`, `Quaternion` from Task 1.
- Produces: frozen `VehicleState`, `VehicleControl`, `VehicleParameters`, and `SurfaceContact`.

- [ ] **Step 1:** Write failing tests for valid construction, derived `VehicleState.speed`, normalized-control validation, parameter validation including positive ride height, quaternion/state immutability, and unit-normal enforcement for `SurfaceContact`.
- [ ] **Step 2:** Run the two focused test files and verify failure because domain APIs do not exist.
- [ ] **Step 3:** Implement the frozen dataclasses exactly as defined by the spec.
- [ ] **Step 4:** Run focused tests and then the full suite; require zero failures.
- [ ] **Step 5:** Commit as `feat(sandbox): add 3d vehicle domain model`.

### Task 3: Ground dynamics

**Files:**
- Create: `f1rl/sandbox/dynamics/__init__.py`
- Create: `f1rl/sandbox/dynamics/ground_dynamics.py`
- Create: `tests/sandbox/test_ground_dynamics.py`

**Interfaces:**
- Consumes: vehicle domain values, `SurfaceContact`, geometry primitives.
- Produces: grounded acceleration, steering angular velocity, tangent-constrained velocity/orientation data required by the coordinator.

- [ ] **Step 1:** Write failing tests for flat-ground acceleration, braking/drag, steering sign, speed cap, slope-aligned forward motion, gravity projected along a slope, and deterministic repeated calls.
- [ ] **Step 2:** Run `python -m pytest tests/sandbox/test_ground_dynamics.py -q` and verify failure because ground dynamics do not exist.
- [ ] **Step 3:** Implement the minimal grounded model from the spec without track or renderer dependencies.
- [ ] **Step 4:** Run focused and full tests; require zero failures.
- [ ] **Step 5:** Commit as `feat(sandbox): implement grounded vehicle dynamics`.
### Task 4: Air dynamics

**Files:**
- Create: `f1rl/sandbox/dynamics/air_dynamics.py`
- Create: `tests/sandbox/test_air_dynamics.py`

**Interfaces:**
- Consumes: vehicle domain values and geometry primitives.
- Produces: airborne linear acceleration and preserved angular velocity for integration.

- [ ] **Step 1:** Write failing tests for gravity, linear air drag, preservation of full 3D velocity, persistence of angular velocity, controls having no direct airborne effect, and deterministic repeated calls.
- [ ] **Step 2:** Run `python -m pytest tests/sandbox/test_air_dynamics.py -q` and verify failure because air dynamics do not exist.
- [ ] **Step 3:** Implement the minimal ballistic-air model from the spec.
- [ ] **Step 4:** Run focused and full tests; require zero failures.
- [ ] **Step 5:** Commit as `feat(sandbox): implement airborne vehicle dynamics`.

### Task 5: Dynamics coordinator and integration

**Files:**
- Create: `f1rl/sandbox/dynamics/vehicle_dynamics.py`
- Create: `f1rl/sandbox/dynamics/integrator.py`
- Create: `tests/sandbox/test_vehicle_dynamics.py`
- Create: `tests/sandbox/test_integrator.py`

**Interfaces:**
- Consumes: outputs and domain values from Tasks 1-4 plus optional `SurfaceContact`.
- Produces: one high-level derivative API and `integrate(...) -> VehicleState` using deterministic semi-implicit Euler integration.

- [ ] **Step 1:** Write failing coordinator tests proving contact selects ground dynamics and `None` selects air dynamics.
- [ ] **Step 2:** Write failing integrator tests for 3D velocity/position advancement, quaternion orientation advancement, non-positive-`dt` validation, grounded tangent constraint without losing tangential displacement, contact loss preserving ramp launch velocity, and landing removing inward normal velocity without bounce.
- [ ] **Step 3:** Run the focused tests and verify expected failures because coordinator/integrator APIs do not exist.
- [ ] **Step 4:** Implement coordinator and integrator exactly to the spec; do not embed track queries inside physics.
- [ ] **Step 5:** Run focused and full tests; require zero failures.
- [ ] **Step 6:** Commit as `feat(sandbox): integrate ground and air physics`.
### Task 6: Physics-core acceptance pass

**Files:**
- Modify only if verification exposes a defect in files created by Tasks 1-5.

**Interfaces:**
- Consumes: all physics-core interfaces.
- Produces: a verified 3D physics baseline ready for track geometry, collision, and perception work.

- [ ] **Step 1:** Add an integration test with a synthetic inclined ramp contact, then remove contact and verify the vehicle becomes airborne with positive vertical velocity.
- [ ] **Step 2:** Continue the same trajectory under gravity and verify vertical velocity decreases deterministically.
- [ ] **Step 3:** Supply a landing contact and verify inward normal velocity is removed while tangential velocity is preserved.
- [ ] **Step 4:** Run `python -m pytest -q`; require zero failures.
- [ ] **Step 5:** Run a repeated-simulation smoke test and assert exact equality for identical inputs.
- [ ] **Step 6:** Run `git diff --check` and `git status --short`; require no whitespace errors and only intended changes.
