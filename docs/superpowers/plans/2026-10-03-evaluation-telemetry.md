# Evaluation Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP4 raw evaluation telemetry in `F1RLEnv.info["telemetry"]` so WP5 can compute episode metrics without changing the frozen PPO observation/action/reward/race protocol.

**Architecture:** Keep telemetry at the Gym adapter boundary. Read accepted post-step vehicle state and actual world events from `SimulationSnapshot`; capture the pre-step `TrackQuery` only to identify boost/slow modifiers that actually influenced the current physics tick. Preserve all existing top-level `info` keys, sampled-track behavior, and SB3 VecEnv semantics.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 2.9.0, NumPy, immutable sandbox/runtime domain contracts, pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-evaluation-telemetry-design.md`

## Global Constraints

- Branch: `research/evaluation-telemetry`.
- Base HEAD: `ea8465311de4ae883b38d0cd9faaba43a5f02401`.
- Approved spec commit: `74ae9083c9b63117134674a5f3c6eea0794ad673`.
- Production telemetry changes are expected only in `f1rl/learning/gym_environment.py`.
- Do not change `f1rl/sandbox/**`, `f1rl/runtime/**`, track assets, experiment YAML, PPO, training runner, reward implementation, or observation implementation.
- Telemetry remains evaluator-facing `info`; it must never enter the policy observation.
- The telemetry mapping contains exactly seven fields from the approved spec.
- No public telemetry dataclass/module, recorder, persistence layer, metric aggregation, or evaluator belongs in WP4.
- No production mocks/fakes/test doubles.
- Direct SB3 production imports remain owned only by `algorithms/ppo.py` and `training/vector_env.py`; direct production Torch imports remain zero.
## Implementation Rulings

### Ruling 1 — VecEnv reset transport

`DummyVecEnv.reset()` returns only batched observations. Per-environment reset info is exposed through `vector.reset_infos`.

Therefore WP4 VecEnv tests must inspect:
- `vector.reset_infos` for reset telemetry;
- `infos` returned by `vector.step(...)` for transition telemetry;
- terminal `infos` for terminal telemetry and `terminal_observation`.

This is an SB3 transport detail, not a change to the approved `F1RLEnv.reset() -> (observation, info)` contract.

### Ruling 2 — No duplicate physics interpretation

The Gym adapter may inspect:
- accepted `SimulationSnapshot.world.vehicle`;
- `SimulationSnapshot.events`;
- the exact pre-step `SimulationSnapshot.query`.

It must not recompute collision detection, modifier strengths, or spatial containment independently. IDs are evidence describing physics decisions already made by sandbox/runtime.

## Review Focus

1. A vehicle entering a modifier during a tick must not report that modifier until the following tick.
2. A vehicle leaving a modifier during a tick must still report it for that tick if it was active at tick start.
3. Spatial overlap while `query_before.contact is None` must report no applied boost/slow IDs.
4. Obstacle telemetry must come from real `ObstacleHit` events, including repeated and multi-hit ticks.
5. Collision physical telemetry must describe the accepted rollback/stopped vehicle, not the rejected candidate.
6. Absolute position and entity IDs must remain absent from the 16-float PPO observation.
7. DummyVecEnv terminal auto-reset must not overwrite terminal telemetry with next-episode reset telemetry.

---

### Task 1: Add the exact reset and post-step physical telemetry contract
**Files:**
- Modify: `tests/learning/test_gym_environment.py`
- Modify: `f1rl/learning/gym_environment.py`

**Interfaces:**
- Consumes: existing `SimulationSnapshot`, `VehicleState`, `F1RLEnv.reset()`, and `F1RLEnv.step()`.
- Produces: nested `info["telemetry"]` with the exact approved seven-field schema.

- [ ] **Step 1: Add failing reset-schema tests**

Extend `test_gym_environment.py` to assert every successful reset returns:

```python
info["telemetry"] == {
    "vehicle_position": (0.0, 0.0, 0.0),
    "vehicle_velocity": (0.0, 0.0, 0.0),
    "vehicle_speed": 0.0,
    "grounded": False,
    "obstacle_hit_ids": (),
    "applied_boost_pad_ids": (),
    "applied_slow_zone_ids": (),
}
```

Also assert:
- `set(telemetry) ==` the exact seven keys;
- vector tuples contain plain `float` values, not `Vec3`;
- `vehicle_speed` is `float`;
- `grounded` is `bool`;
- repeated resets and different Gym seeds preserve deterministic telemetry.

- [ ] **Step 2: Run the focused reset test and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/test_gym_environment.py -q
```

Expected: reset-info assertions fail because telemetry does not exist yet.
- [ ] **Step 3: Implement the minimum private telemetry helpers**

In `f1rl/learning/gym_environment.py`, add private helper logic that:
- converts `VehicleState.position` and `velocity` to explicit 3-tuples of `float`;
- uses `float(vehicle.speed)`;
- uses the vehicle's actual `grounded` bool;
- accepts explicit event/modifier ID tuples rather than holding mutable state.

Do not export helpers from `f1rl.learning`.

- [ ] **Step 4: Add reset telemetry only**

Make `_reset_info()` include `"telemetry"` built from the current reset snapshot with all three interaction/modifier tuples empty.

Do not infer modifiers from reset `snapshot.query`, even if spawn lies inside a zone.

- [ ] **Step 5: Add failing post-step physical-state tests**

Add tests proving after one valid step:
- telemetry position equals `snapshot_after.world.vehicle.position`;
- telemetry velocity equals `snapshot_after.world.vehicle.velocity`;
- telemetry speed equals `snapshot_after.world.vehicle.speed`;
- `speed == sqrt(vx**2 + vy**2 + vz**2)`;
- grounded equals accepted post-step vehicle grounded state.

For this Task, interaction tuples may remain empty except where later tasks specialize them.

- [ ] **Step 6: Wire physical telemetry into step info**

Capture the accepted `snapshot_after` already returned by `Simulation.step()` and include physical telemetry in `_step_info(...)`.

Do not alter reward evaluation, race-progress advancement, termination/truncation, or observation encoding order.

- [ ] **Step 7: Run Task 1 tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/test_gym_environment.py -q
```

Expected: PASS.
- [ ] **Step 8: Commit Task 1**

```powershell
git add f1rl/learning/gym_environment.py tests/learning/test_gym_environment.py
git commit -m "feat(evaluation): expose vehicle telemetry"
```

---

### Task 2: Pin actual obstacle and applied modifier semantics

**Files:**
- Modify: `tests/learning/test_gym_environment.py`
- Create: `tests/integration/test_evaluation_telemetry.py`
- Modify: `f1rl/learning/gym_environment.py`

**Interfaces:**
- Consumes: `SimulationSnapshot.events`, pre-step `SimulationSnapshot.query`, canonical `slalom-v1` / `right-turn-v1`, existing WP1B environment contract.
- Produces: event/modifier ID tuples describing what actually happened in each tick.

- [ ] **Step 1: Add failing obstacle-event unit coverage**

Construct real track/world conditions in the test file so an accepted Gym step produces an `ObstacleHit`.

Assert:
- `obstacle_hit_ids` contains the exact obstacle ID;
- telemetry physical state is the accepted rollback position;
- velocity and speed are zero after the blocker rejects candidate motion;
- a repeated physical hit on a later tick produces the same ID again;
- if one tick genuinely produces multiple obstacle events, all IDs are retained in event order.

Do not infer collision from reward or zero velocity in the assertions.

- [ ] **Step 2: Implement obstacle extraction from events**

Extract IDs only from `ObstacleHit` instances in the current `snapshot_after.events`.

Preserve event order; do not sort obstacle IDs again in the Gym adapter.
- [ ] **Step 3: Capture the exact pre-step query before simulation mutation**

Immediately before:

```python
snapshot_after = self._simulation.step(control)
```

capture:

```python
query_before = self._simulation.snapshot.query
```

Pass that query into private step-telemetry construction after a successful simulation step.

Do not use `snapshot_after.query` to determine applied modifier IDs.

- [ ] **Step 4: Add canonical modifier integration tests and verify RED**

Create `tests/integration/test_evaluation_telemetry.py` using real canonical assets and the checked-in WP1B experiment values.

Pin:
- grounded tick beginning inside slalom `boost-0` -> `("boost-0",)`;
- grounded tick beginning inside right-turn `slow-0` -> `("slow-0",)`;
- modifier tuples are lexicographically sorted when multiple entities are present;
- no-contact/airborne query -> both applied modifier tuples empty even if the position overlaps a modifier volume.

Use test-only construction helpers in the integration test file; do not add a production environment-builder solely for WP4.

- [ ] **Step 5: Add zone entry/exit timing integration coverage**

Choose boundary-adjacent grounded canonical states and first prove with `TrackWorld.query_point(...)` which side of the modifier boundary each pre/post state occupies.

Assert:
- outside at tick start -> enters during tick -> current telemetry does not list modifier;
- next tick starting inside -> modifier is listed;
- inside at tick start -> leaves during tick -> current telemetry still lists modifier.

Tests must validate their setup via queries so they cannot pass because a chosen speed/position accidentally missed the zone.

- [ ] **Step 6: Implement applied modifier extraction**

For `query_before.contact is None`:
```python
applied_boost_pad_ids = ()
applied_slow_zone_ids = ()
```

Otherwise use lexicographically sorted IDs from `query_before.boost_pads` and `query_before.slow_zones`.
- [ ] **Step 7: Run Task 2 focused tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/test_gym_environment.py tests/integration/test_evaluation_telemetry.py tests/integration/test_canonical_track_mechanics.py -q
```

Expected: PASS.

- [ ] **Step 8: Verify sandbox/runtime remain unchanged**

```powershell
git diff --exit-code ea8465311de4ae883b38d0cd9faaba43a5f02401 -- f1rl/sandbox f1rl/runtime
```

Expected: no diff.

- [ ] **Step 9: Commit Task 2**

```powershell
git add f1rl/learning/gym_environment.py tests/learning/test_gym_environment.py tests/integration/test_evaluation_telemetry.py
git commit -m "feat(evaluation): report interaction telemetry"
```

---

### Task 3: Prove sampled-wrapper and DummyVecEnv preservation

**Files:**
- Modify: `tests/learning/training/test_vector_env.py`
- No production vector-env change expected.

**Interfaces:**
- Consumes: existing `_SampledF1RLEnv`, `make_vector_env()`, `make_multi_track_vector_env()`, SB3 `DummyVecEnv`.
- Produces: regression proof that nested telemetry survives reset, step, sampling, and terminal auto-reset.

- [ ] **Step 1: Add standard DummyVecEnv reset/step assertions**

For a standard dummy vector:
- after `vector.reset()`, assert every `vector.reset_infos[i]["telemetry"]` has the exact seven keys;
- after `vector.step(...)`, assert every returned `infos[i]["telemetry"]` is the current transition telemetry.
- [ ] **Step 2: Add sampled-wrapper preservation assertions**

Directly exercise `_SampledF1RLEnv` and assert on reset and step:
- nested `telemetry` is unchanged from the active inner `F1RLEnv`;
- top-level `track_id` is present alongside telemetry;
- scenario sampling behavior and RNG stream remain unchanged.

Do not change selection seed derivation or reset-seed semantics.

- [ ] **Step 3: Add terminal auto-reset regression**

Use `make_multi_track_vector_env(...)` with `VectorEnvConfig(backend="dummy")` and an episode that terminates/truncates predictably.

Before the terminal step, record each active scenario ID.

After terminal `vector.step(...)`, assert each terminal `info` contains:
- the previous episode's `track_id`;
- the previous episode's terminal `telemetry`;
- `terminal_observation`.

Also assert the slot has already reset to a next episode, while the returned terminal telemetry has not been replaced by `vector.reset_infos` for that next episode.

- [ ] **Step 4: Run vector tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/training/test_vector_env.py -q
```

Expected: PASS without production changes to `training/vector_env.py`.

- [ ] **Step 5: If and only if a real SB3 compatibility defect appears**

Add a focused regression test first, then make the smallest vector-env repair needed.

Do not preemptively modify vector production code.

- [ ] **Step 6: Commit Task 3**

If tests only:
```powershell
git add tests/learning/training/test_vector_env.py
git commit -m "test(evaluation): preserve telemetry through vecenv"
```

If a real production fix was required, include only the demonstrated minimal fix in the same task commit and document the reason in the commit body.
---

### Task 4: Protocol no-leak audit and exact WP4 verification

**Files:**
- Modify only if a regression test exposes a real WP4 defect.
- No feature expansion is expected.

**Interfaces:**
- Consumes: all WP4 commits from Tasks 1–3.
- Produces: exact verification evidence and a clean branch ready for the integration decision.

- [ ] **Step 1: Add/strengthen frozen protocol regression**

Using the checked-in canonical experiment and real `F1RLEnv`, assert:
- canonical observation size remains exactly 16;
- reset observation is byte/value-identical to the pre-WP4 expected protocol;
- observation-space low/high/shape remain unchanged;
- action-space low/high/shape remain unchanged;
- identical transition inputs produce the same reward total/components as before telemetry;
- existing race-progress fields retain their semantics.

The only new Gym-return semantic is nested `info["telemetry"]`.

- [ ] **Step 2: Verify privileged telemetry cannot reach observation construction**

Review the full WP4 diff and require:
- no changes in `f1rl/learning/observation.py`;
- no telemetry argument added to `ObservationEncoder.encode(...)`;
- no absolute position or entity IDs inserted into observation arrays;
- no changes to experiment observation config.

- [ ] **Step 3: Run focused WP4 suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning/test_gym_environment.py tests/learning/training/test_vector_env.py tests/integration/test_evaluation_telemetry.py tests/integration/test_canonical_track_mechanics.py tests/learning/experiments/test_environment_protocol.py -q
```

Expected: PASS.

- [ ] **Step 4: Run learning/orchestration regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/learning tests/integration/test_training_orchestration_integration.py tests/integration/test_vectorized_ppo_integration.py -q
```

Expected: PASS.
- [ ] **Step 5: Run the full/static/dependency gate**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q f1rl scripts
.\.venv\Scripts\python.exe -m pip check
git diff --check ea8465311de4ae883b38d0cd9faaba43a5f02401...HEAD
```

Audit and require:
- direct production SB3 import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`;
- direct production `torch` imports remain zero;
- production Mock/Fake/test-double markers remain zero;
- no diff under `f1rl/sandbox/**`, `f1rl/runtime/**`, `assets/tracks/**`, or experiment YAML;
- no new public telemetry exports.

- [ ] **Step 6: Whole-WP4 review**

Review `ea8465311de4ae883b38d0cd9faaba43a5f02401...HEAD` with priority on:
1. observation/policy information leakage;
2. post-step query incorrectly used for applied modifier telemetry;
3. collision IDs inferred rather than event-derived;
4. candidate state leaking instead of accepted vehicle state;
5. terminal telemetry overwritten by DummyVecEnv auto-reset;
6. accidental metric computation or WP5 scope creep;
7. sandbox/runtime/track/protocol drift.

Any Critical/Important defect gets a dedicated regression test before repair.

- [ ] **Step 7: Exact-final rerun**

After any review fixes, rerun Steps 3–5 on the exact final HEAD.

Record:
- final HEAD;
- commit list;
- focused/full test counts;
- compileall result;
- pip check result;
- diff-check result;
- dependency-boundary result;
- final `git status`.

- [ ] **Step 8: Final branch state and stop**

Require:

```text
branch == research/evaluation-telemetry
git status --porcelain == empty
all verification commands green
no merge/push
WP5 not started
```

Stop at the WP4 integration gate and ask the user whether to merge, push, or keep the branch/worktree.

## Stop Condition

After Task 4, **STOP**. Do not begin WP5 Evaluator + Metrics automatically.
