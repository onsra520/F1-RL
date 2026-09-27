# Composable Transition Rewards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a deterministic, composable V1 reward system for one accepted simulation transition with inspectable per-term contributions.

**Architecture:** `RewardTransition` carries immutable before/after vehicle state, one fixed pre-step target, accepted world events, and `dt`. Stateless built-in terms evaluate that transition, while `RewardComposer` preserves term order and produces an immutable `RewardResult` without clipping, normalization, runtime ownership, or route-progress inference.

**Tech Stack:** Python 3.12, dataclasses, `typing.Protocol`, pytest, existing F1-RL sandbox/runtime test fixtures.

**Spec:** `docs/superpowers/specs/2026-09-28-composable-transition-reward-design.md`

## Global Constraints

- Production reward code may depend only on Python stdlib and sandbox domain/event types.
- No production dependency on runtime, API, drivers, observation, action, Gymnasium, NumPy, PyTorch, SB3/PPO, FastAPI/WebSocket, React, Three.js, randomness, wall clock, network, or mutable global state.
- Game score remains separate from RL reward; no implicit `ScoreState` or pickup-point mapping.
- Checkpoint and finish entry events are not rewarded in V1.
- All reward configuration is finite numeric `>= 0.0`; booleans are rejected; `0.0` disables a term.
- Reward evaluation performs no clipping, normalization, target mutation, simulation stepping, or episode/progress ownership.

## Review Focus

- `RewardTransition.events` containing any non-WorldEvent object must fail at construction rather than being silently ignored -- pinned in Task 1.
- Zero-valued term configuration must remain valid and yield exact zero contributions -- pinned in Task 2.
- Multiple `ObstacleHit` events in one transition must still produce exactly one collision penalty -- pinned in Task 2.
- A term returning a mismatched contribution name or non-finite value must be rejected by the composer -- pinned in Task 3.
- Checkpoint, finish, and score-pickup events must not leak implicit reward through built-in terms or composition -- pinned in Tasks 2 and 4.

## File Structure

- Create `f1rl/learning/rewards/core.py` -- transition/result DTOs, `RewardTerm` protocol, and composer.
- Create `f1rl/learning/rewards/terms.py` -- V1 built-in stateless reward terms.
- Create `f1rl/learning/rewards/__init__.py` -- public reward package exports.
- Modify `f1rl/learning/rewards/README.md` -- replace stale planned-file layout with implemented V1 contract.
- Modify f1rl/learning/README.md -- document reward V1 alongside observation/action baselines.
- Modify `f1rl/learning/__init__.py` -- re-export reward APIs while preserving observation/action exports.
- Create focused unit and integration tests under `tests/learning/rewards/` and `tests/integration/`.

---

### Task 1: Reward transition and result contracts

**Files:**
- Create: `f1rl/learning/rewards/core.py`
- Create: `f1rl/learning/rewards/__init__.py`
- Create: `tests/learning/rewards/test_reward_contract.py`

**Interfaces:**
- Consumes: sandbox `VehicleState`, `Vec3`, and current concrete WorldEvent variants.
- Produces: `RewardTransition`, `RewardContribution`, `RewardResult`, and `RewardTerm`.

- [ ] **Step 1: Write failing `RewardTransition` validation tests**

Cover valid `VehicleState`/`Vec3|None`/event tuple/positive `dt`; reject wrong vehicle or target types, non-tuple events, an event tuple containing a non-WorldEvent object, booleans/non-numeric/non-finite `dt`, and `dt <= 0.0`. Assert the dataclass is frozen.

- [ ] **Step 2: Write failing contribution/result validation tests**

Assert `RewardContribution("name", finite_value)` works; reject blank/non-string names and non-finite values. Assert `RewardResult(finite_total, tuple)` works; reject non-finite total or non-tuple contributions and prove result immutability.

- [ ] **Step 3: Run RED**

Run: `python -m pytest tests/learning/rewards/test_reward_contract.py -q`

Expected: import/collection failure because `f1rl.learning.rewards.core` does not exist.

- [ ] **Step 4: Implement immutable contracts and protocol**

In `core.py`, implement the three dataclasses with exact spec validation plus:
`class RewardTerm(Protocol)` exposing `name: str` and `evaluate(transition: RewardTransition) -> RewardContribution`.
Validate events against the current concrete variants `ObstacleHit`, `CheckpointEntered`, `FinishLineEntered`, and `ScorePickupCollected`; do not infer future variants dynamically.

- [ ] **Step 5: Export Task 1 reward APIs from package**

Create `f1rl/learning/rewards/__init__.py` exporting `RewardTransition`, `RewardContribution`, `RewardResult`, and `RewardTerm` only for now.

- [ ] **Step 6: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/rewards/test_reward_contract.py -q`
- `python -m pytest tests/learning -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 7: Commit**

Commit message: `feat(learning): add reward transition contracts`

---

### Task 2: Built-in V1 reward terms

**Files:**
- Create: `f1rl/learning/rewards/terms.py`
- Modify: `f1rl/learning/rewards/__init__.py`
- Create: `tests/learning/rewards/test_reward_terms.py`

**Interfaces:**
- Consumes: Task 1 `RewardTransition` and `RewardContribution`.
- Produces: `TargetProgressReward`, `CollisionPenalty`, and `TimePenalty`.

- [ ] **Step 1: Write failing configuration-validation tests**

For each built-in term, accept `0.0` and ordinary finite non-negative values; reject booleans, strings/objects, `nan`, infinities, and negative values.

- [ ] **Step 2: Write failing target-progress tests**

Assert name is exactly `"target_progress"`; `target=None` returns exact zero; moving from 10m to 8m with weight `2.0` yields `+4.0`; moving away yields a signed negative value; equal distance yields zero. Include non-axis-aligned 3D positions to prove Euclidean distance rather than a single-axis shortcut.

- [ ] **Step 3: Write failing collision tests**

Assert name is exactly `"collision"`; zero obstacle events yields `0.0`; one or several `ObstacleHit` events yield exactly `-penalty` once. Mix checkpoint, finish, and score-pickup events and prove they do not change the collision contribution.

- [ ] **Step 4: Write failing time tests**

Assert name is exactly `"time"` and `TimePenalty(0.1)` yields `-0.01` for `dt=0.1` and `-0.005` for `dt=0.05`. Verify zero configuration yields exact zero.

- [ ] **Step 5: Run RED**

Run: `python -m pytest tests/learning/rewards/test_reward_terms.py -q`

Expected: import failure because built-in terms do not yet exist.

- [ ] **Step 6: Implement the three stateless terms**

Use one shared private finite non-negative config validator. Each term returns a `RewardContribution` with its fixed analytics name; perform no clipping, target switching, score lookup, checkpoint interpretation, or hidden state.

- [ ] **Step 7: Export built-in terms from reward package**

Update `f1rl/learning/rewards/__init__.py` to export the three term classes while preserving Task 1 exports.

- [ ] **Step 8: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/rewards/test_reward_terms.py -q`
- `python -m pytest tests/learning/rewards -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 9: Commit**

Commit message: `feat(learning): add transition reward terms`

---

### Task 3: Ordered reward composer and breakdown

**Files:**
- Modify: `f1rl/learning/rewards/core.py`
- Modify: `f1rl/learning/rewards/__init__.py`
- Create: `tests/learning/rewards/test_reward_composer.py`

**Interfaces:**
- Consumes: Task 1 contracts and any `RewardTerm` implementation, including Task 2 built-ins.
- Produces: `RewardComposer(terms: tuple[RewardTerm, ...])` with `evaluate(...) -> RewardResult`.

- [ ] **Step 1: Write failing composer-construction tests**

Assert `terms` must be a tuple; empty tuple is valid; each term must expose a non-empty string `name` and callable `evaluate`; duplicate names raise `ValueError` at construction. Use tiny local fake terms only to exercise composer contract.

- [ ] **Step 2: Write failing ordered-evaluation tests**

Use three instrumented fake terms with names in deliberately non-lexical order. Assert evaluation call order exactly matches declaration order, each term is called once, contributions preserve that order, and total is their ordinary un-clipped sum.

- [ ] **Step 3: Write failing invalid-term-output tests**

Assert composer rejects a term that returns a non-`RewardContribution`, a contribution whose name differs from the term's declared name, or a malformed `RewardContribution` carrying a non-finite value. In the test only, synthesize the malformed frozen DTO via `object.__new__`/`object.__setattr__` so the composer's defense-in-depth path is actually exercised. Do not silently rename, clamp, or skip invalid output.

- [ ] **Step 4: Write failing empty-composer and determinism tests**

Assert `RewardComposer(()).evaluate(transition) == RewardResult(0.0, ())`; repeated evaluation with equal immutable transition/configuration returns equal results and leaves composer/transition unchanged.

- [ ] **Step 5: Run RED**

Run: `python -m pytest tests/learning/rewards/test_reward_composer.py -q`

Expected: import/attribute failure because `RewardComposer` does not yet exist.

- [ ] **Step 6: Implement `RewardComposer` in `core.py`**

Validate the tuple and structural term contract at construction; cache no mutable evaluation state. `evaluate()` calls each term once in declaration order, validates each returned contribution against the term name and finite-value contract, then constructs `RewardResult(sum(values), contributions)` with no clipping/normalization.

- [ ] **Step 7: Export composer from reward package**

Update `f1rl/learning/rewards/__init__.py` to export `RewardComposer`.

- [ ] **Step 8: Run GREEN + regression suite**

Run:
- `python -m pytest tests/learning/rewards/test_reward_composer.py -q`
- `python -m pytest tests/learning/rewards -q`
- `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 9: Commit**

Commit message: `feat(learning): compose deterministic reward breakdowns`

---

### Task 4: Simulation integration, public exports, docs, and dependency acceptance

**Files:**
- Create: `tests/integration/test_reward_integration.py`
- Modify: `f1rl/learning/rewards/README.md`
- Modify: f1rl/learning/README.md
- Modify: `f1rl/learning/__init__.py`

**Interfaces:**
- Consumes: complete Tasks 1-3 reward API plus existing runtime `Simulation` in tests only.
- Produces: verified public reward surface ready for future route-progress and Gymnasium layers.

- [ ] **Step 1: Add real forward/away simulation acceptance tests**

Create independent deterministic simulations and capture `before`/`after` snapshots. Build `RewardTransition` with one target captured before each step. Prove a real step toward the target yields positive `target_progress` plus negative time contribution, while an equivalent step oriented/moving away yields negative progress.

- [ ] **Step 2: Add blocked-collision and event-safety acceptance tests**

Build a real step blocked by overlapping obstacle volumes so `after.events` contains multiple `ObstacleHit` values; assert exactly one collision penalty. Separately feed checkpoint, finish, and score-pickup events and prove the standard V1 composer receives no implicit bonus from them beyond configured progress/time semantics.

- [ ] **Step 3: Add deterministic independent-transition acceptance test**

Create two independent but equal before/after transitions and equal composers; assert complete `RewardResult` equality, including ordered contribution breakdowns, and verify source sandbox objects are unchanged.

- [ ] **Step 4: Add public-export acceptance test**

Import all eight reward symbols from both `f1rl.learning.rewards` and `f1rl.learning`; assert identity/value agreement and prove existing action/observation exports remain intact.

- [ ] **Step 5: Add source/dependency boundary checks**

Parse all production `f1rl/learning/rewards/*.py` imports and reject runtime/API/driver/observation/action/Gymnasium/NumPy/PyTorch/SB3/FastAPI/WebSocket/rendering dependencies. Scan `f1rl/sandbox/**/*.py` and prove none import `f1rl.learning`.

- [ ] **Step 6: Update reward and learning README documentation**

Update both `f1rl/learning/rewards/README.md` and `f1rl/learning/README.md`. Replace stale planned reward filenames with the implemented `core.py`/`terms.py` model; document fixed-target transition timing, the three V1 terms, ordered contribution breakdown, game-score separation, and that checkpoint/finish bonuses plus episode semantics remain future work.

- [ ] **Step 7: Re-export complete reward API from `f1rl.learning`**

Preserve all existing action/observation exports while adding `RewardTransition`, `RewardContribution`, `RewardResult`, `RewardTerm`, `RewardComposer`, `TargetProgressReward`, `CollisionPenalty`, and `TimePenalty`.

- [ ] **Step 8: Run final verification**

Run:
- `python -m pytest -q`
- `python -m compileall -q f1rl`
- dependency/source scans from Step 5
- `git diff --check`

Expected: zero test failures, successful compilation, no forbidden dependencies, and clean whitespace checks.

- [ ] **Step 9: Commit**

Commit message: `test(learning): verify composable transition rewards`

## Expected branch result

The feature branch should contain only learning reward code/tests/docs plus this plan/spec. No sandbox or runtime production file should change. V1 remains stateless and framework-agnostic: progress is measured against one fixed pre-step target, collision is penalized once per accepted transition, time pressure scales by seconds, and checkpoint/finish/game-score semantics remain outside reward ownership.
