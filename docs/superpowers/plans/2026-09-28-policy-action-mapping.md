# Strict 2D Policy Action Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic two-dimensional policy action contract that maps validated steering/throttle values exactly into sandbox `VehicleControl` with brake fixed to `0.0`.

**Architecture:** Add a framework-agnostic `f1rl.learning.action` module that owns policy-facing validation, metadata, and exact conversion into `VehicleControl`. Keep runtime, Gymnasium, PPO, NumPy, drivers, rewards, and observation encoding outside the production dependency graph; use runtime only in integration tests to prove semantic equivalence.

**Tech Stack:** Python >=3.11, standard-library `dataclasses`/`math`, existing sandbox `VehicleControl`, pytest >=8,<10.

**Spec:** `docs/superpowers/specs/2026-09-27-policy-action-mapping-design.md`

## Global Constraints

- V1 policy action layout is exactly `(steering, throttle)`.
- `steering` is finite numeric in `[-1.0, 1.0]`; `throttle` is finite numeric in `[0.0, 1.0]`.
- Booleans are rejected even though `bool` subclasses `int`.
- Invalid values fail with `ValueError`; no clamping, wrapping, normalization, or hidden remapping.
- `map_policy_action(...)` copies steering/throttle exactly and always emits `brake == 0.0`.
- `ACTION_NAMES`, `ACTION_LOW`, `ACTION_HIGH`, and `ACTION_SIZE` are the authoritative stable V1 layout.
- `f1rl.learning.action` may depend only on sandbox `VehicleControl` and Python standard library.
- No sandbox or runtime production files may be modified.

## Review Focus

- Boundary values `-1.0`, `0.0`, and `1.0` must remain valid where the contract says inclusive.
- Integer inputs such as `0`/`1` are valid, while `True`/`False` are not.
- Mapper input must be an actual `PolicyAction`, not merely an object with matching attributes.
- Throttle `0.0` must remain exactly `0.0`; no future-adapter-friendly remapping belongs here.
- Runtime-equivalence acceptance must compare two independently constructed simulations so state mutation cannot make the assertion vacuous.

---

### Task 1: Policy action DTO and authoritative metadata

**Files:**
- Create: `f1rl/learning/action.py`
- Modify: `f1rl/learning/__init__.py`
- Create: `tests/learning/test_action_contract.py`

**Interfaces:**
- Consumes: Python numeric primitives only.
- Produces: `PolicyAction`, `ACTION_NAMES`, `ACTION_LOW`, `ACTION_HIGH`, and `ACTION_SIZE`.

- [ ] **Step 1: Write failing metadata tests**

Assert exactly:
`ACTION_NAMES == ("steering", "throttle")`, `ACTION_LOW == (-1.0, 0.0)`, `ACTION_HIGH == (1.0, 1.0)`, and `ACTION_SIZE == 2`. Also assert metadata order matches `PolicyAction` field order.

- [ ] **Step 2: Write failing valid-construction and immutability tests**

Cover exact boundaries `PolicyAction(-1.0, 0.0)` and `PolicyAction(1.0, 1.0)`, ordinary in-range values, valid integer `0`/`1`, and frozen-dataclass mutation rejection.

- [ ] **Step 3: Write failing invalid-value tests**

Parametrize steering/throttle with booleans, strings/objects, `nan`, positive/negative infinity, and values just outside each interval; every invalid construction must raise `ValueError`.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/learning/test_action_contract.py -q`

Expected: import/collection failure because `f1rl.learning.action` does not exist.

- [ ] **Step 5: Implement DTO and metadata**

In `f1rl/learning/action.py`, add frozen/slots `PolicyAction(steering: float, throttle: float)` with explicit finite numeric validation and inclusive bounds. Define the four exact tuple/size constants from the spec. Do not coerce or clamp stored values.

Update `f1rl/learning/__init__.py` to export the new action contract without removing observation exports.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/test_action_contract.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(learning): define policy action contract`

---

### Task 2: Exact policy-to-vehicle control mapping

**Files:**
- Modify: `f1rl/learning/action.py`
- Modify: `f1rl/learning/__init__.py`
- Create: `tests/learning/test_action_mapping.py`

**Interfaces:**
- Consumes: Task 1 `PolicyAction` and existing `VehicleControl`.
- Produces: `map_policy_action(action: PolicyAction) -> VehicleControl`.

- [ ] **Step 1: Write failing exact-mapping tests**

For representative actions including `(-1.0, 0.0)`, `(0.0, 0.5)`, and `(1.0, 1.0)`, assert mapped control steering/throttle equal the supplied values exactly and `brake == 0.0` exactly. Explicitly prove throttle `0.0` does not become `0.5`.

- [ ] **Step 2: Write failing strict-input-type test**

Pass `None`, tuples, dicts, and a lookalike object with `steering`/`throttle` attributes; each must raise `ValueError` because the mapper requires an actual `PolicyAction`.

- [ ] **Step 3: Write failing determinism/immutability test**

Map the same immutable action repeatedly and through an equal copy; assert equal `VehicleControl` values and unchanged input action.

- [ ] **Step 4: Run RED**

Run: `python -m pytest tests/learning/test_action_mapping.py -q`

Expected: import/attribute failure because `map_policy_action(...)` does not exist.

- [ ] **Step 5: Implement exact mapper**

Add `map_policy_action(action: PolicyAction) -> VehicleControl` in `f1rl/learning/action.py`. Reject non-`PolicyAction` inputs with `ValueError`; otherwise construct `VehicleControl(action.steering, action.throttle, 0.0)` with no additional transform.

Export `map_policy_action` from `f1rl.learning`.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/test_action_mapping.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(learning): map policy actions to vehicle controls`

---

### Task 3: Runtime equivalence and dependency acceptance

**Files:**
- Create: `tests/integration/test_action_mapping.py`
- Modify: `f1rl/learning/README.md`

**Interfaces:**
- Consumes: complete Tasks 1-2 action APIs plus existing sandbox world/vehicle types and runtime `Simulation` in tests only.
- Produces: a verified action baseline ready for a future Gymnasium adapter.

- [ ] **Step 1: Add one-step runtime-equivalence test**

Construct two independent but equal `Simulation` instances with the same `TrackWorld`, `WorldState`, `VehicleParameters`, and `SimulationConfig`. Step one with `map_policy_action(PolicyAction(steering, throttle))` and the other with `VehicleControl(steering, throttle, 0.0)`; assert complete `SimulationSnapshot` equality.

Use non-zero steering and throttle so the test exercises both mapped fields rather than a neutral-control identity case.

- [ ] **Step 2: Add public-export acceptance test**

Import all six action symbols from `f1rl.learning` and assert their identities/values match the authoritative `f1rl.learning.action` definitions without disturbing existing observation exports.

- [ ] **Step 3: Add source/dependency boundary checks**

Parse `f1rl/learning/action.py` imports and reject runtime/API/driver/observation/reward/Gymnasium/NumPy/PyTorch/SB3/PPO/FastAPI/WebSocket/rendering dependencies. Scan `f1rl/sandbox/**/*.py` and prove none import `f1rl.learning`.

- [ ] **Step 4: Update learning README**

Document the implemented strict 2D action contract, exact metadata bounds/order, fail-fast validation, exact no-remap mapping, fixed `brake=0.0`, and the fact that Gymnasium/PPO/braking remain future layers.

- [ ] **Step 5: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- dependency/source scans from Step 3
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden dependencies, and clean whitespace checks.

- [ ] **Step 6: Commit**

Commit message: `test(learning): verify deterministic action mapping`

## Expected branch result

The feature branch should contain only learning-layer action code/tests/docs plus this plan/spec. No sandbox or runtime production file should change. The final action contract remains two-dimensional and framework-agnostic, with one-step runtime equivalence proving that the mapper is semantically transparent.
