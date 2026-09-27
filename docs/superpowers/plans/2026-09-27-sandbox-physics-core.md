# Sandbox Physics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic 2D vehicle physics kernel for F1-RL that turns normalized controls into the next `VehicleState`.

**Architecture:** Keep vehicle domain values in `vehicle.py`; compute pure motion derivatives in `vehicle_dynamics.py`; advance state in `integrator.py`. The sandbox remains independent of RL, API, rendering, and track logic.

**Tech Stack:** Python 3.11+, standard-library `dataclasses` and `math`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-sandbox-physics-core-design.md`

## Global Constraints

- `f1rl.sandbox` is deterministic and presentation/RL independent.
- Coordinates are 2D `(x, y)` with heading in radians; heading `0` points along `+x` and increases counter-clockwise.
- Controls are normalized: steering `[-1, 1]`, throttle/brake `[0, 1]`.
- V1 is forward-only; integrated speed is clamped to `[0, max_speed]`.
- Use a kinematic bicycle approximation and explicit Euler integration only.
- No third-party runtime dependencies are introduced in this milestone.

## Review Focus

- Invalid normalized controls must raise rather than silently clip.
- Invalid physical parameters, especially zero wheelbase and steering angle near `pi/2`, must raise.
- Braking must never integrate into negative speed.
- Heading normalization must be stable around the `-pi/pi` boundary.
- Physics functions must not mutate their input dataclasses.

---

### Task 1: Vehicle domain values

**Files:**
- Create: `f1rl/__init__.py`
- Create: `f1rl/sandbox/__init__.py`
- Create: `f1rl/sandbox/vehicle.py`
- Create: `tests/sandbox/test_vehicle.py`

**Interfaces:**
- Consumes: none.
- Produces: `VehicleState`, `VehicleControl`, and `VehicleParameters` frozen dataclasses with constructor validation.

- [ ] **Step 1: Write failing validation and immutability tests** for valid construction, negative speed, out-of-range controls, invalid wheelbase, invalid steering angle, negative physical limits, and frozen instances.
- [ ] **Step 2: Run `python -m pytest tests/sandbox/test_vehicle.py -q`** and verify failure because the domain classes do not exist.
- [ ] **Step 3: Implement the three dataclasses** exactly as defined by the spec, with validation in `__post_init__`.
- [ ] **Step 4: Run the focused test and then `python -m pytest -q`**; both must pass.
- [ ] **Step 5: Commit** as `feat(sandbox): add vehicle domain model`.

### Task 2: Vehicle dynamics derivatives

**Files:**
- Create: `f1rl/sandbox/dynamics/__init__.py`
- Create: `f1rl/sandbox/dynamics/vehicle_dynamics.py`
- Create: `tests/sandbox/test_vehicle_dynamics.py`

**Interfaces:**
- Consumes: `VehicleState`, `VehicleControl`, `VehicleParameters` from Task 1.
- Produces: frozen `MotionDerivatives` and `compute_derivatives(state, control, parameters) -> MotionDerivatives`.

- [ ] **Step 1: Write failing tests** proving straight-line derivatives, throttle acceleration, braking/drag deceleration, positive/negative steering yaw-rate sign, and repeat-call determinism.
- [ ] **Step 2: Run `python -m pytest tests/sandbox/test_vehicle_dynamics.py -q`** and verify failure because the dynamics API does not exist.
- [ ] **Step 3: Implement `MotionDerivatives` and `compute_derivatives`** using the exact equations from the spec.
- [ ] **Step 4: Run the focused test and then `python -m pytest -q`**; both must pass.
- [ ] **Step 5: Commit** as `feat(sandbox): implement vehicle dynamics`.

### Task 3: State integration

**Files:**
- Create: `f1rl/sandbox/dynamics/integrator.py`
- Create: `tests/sandbox/test_integrator.py`

**Interfaces:**
- Consumes: `VehicleState`, `VehicleParameters`, and `MotionDerivatives` from Tasks 1-2.
- Produces: `integrate(state, derivatives, parameters, dt) -> VehicleState`.

- [ ] **Step 1: Write failing tests** for Euler position/heading/speed advancement, speed floor at zero, speed cap, positive-`dt` validation, heading normalization, input immutability, and deterministic repeated integration.
- [ ] **Step 2: Run `python -m pytest tests/sandbox/test_integrator.py -q`** and verify failure because `integrate` does not exist.
- [ ] **Step 3: Implement `integrate`** with explicit Euler, speed clamping, and heading normalization to `[-pi, pi)`.
- [ ] **Step 4: Run the focused test and then `python -m pytest -q`**; both must pass.
- [ ] **Step 5: Commit** as `feat(sandbox): add deterministic state integrator`.

### Task 4: Physics-core acceptance pass

**Files:**
- Modify only if verification exposes a defect in files created by Tasks 1-3.

**Interfaces:**
- Consumes: all physics-core interfaces.
- Produces: a verified baseline ready for world/collision/perception work.

- [ ] **Step 1: Run `python -m pytest -q`** and require zero failures.
- [ ] **Step 2: Run a small one-process deterministic smoke check** that computes derivatives and integrates the same initial values twice, asserting exact dataclass equality.
- [ ] **Step 3: Run `git diff --check` and `git status --short`**; require no whitespace errors and only intended changes before the final commit/push state.
