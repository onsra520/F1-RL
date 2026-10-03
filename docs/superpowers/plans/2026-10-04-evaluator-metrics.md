# Evaluator + Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP5 as a deterministic single-episode evaluator plus immutable episode/per-track metric contracts, using only normal `F1RLEnv` outputs and WP4 telemetry, without training models or inspecting held-out performance.

**Architecture:** Add one small `f1rl.learning.evaluation` package. `metrics.py` owns immutable result DTOs, normalized route-progress math, and per-track descriptive aggregation. `evaluator.py` owns the structural policy protocol, one direct `F1RLEnvFactory` rollout, evidence validation, and step-by-step accumulation. Existing `PPOModel` remains unchanged and satisfies the policy protocol structurally.

**Tech Stack:** Python 3.12, NumPy, Gymnasium contracts through existing `F1RLEnv`, project dataclasses, pytest. No direct Stable-Baselines3 or Torch import in the evaluation package.

**Spec:** `docs/superpowers/specs/2026-10-03-evaluator-metrics-design.md`

## Global Constraints

- Branch: `research/evaluator-metrics`.
- Base HEAD: `032194d470b914c9e0d10fda8c19d5cfe6820630`.
- Approved spec commit: `e2653a817e40ccdc922c1025b5d59d8553e6f572`.
- New production surface is limited to:
  - `f1rl/learning/evaluation/__init__.py`
  - `f1rl/learning/evaluation/metrics.py`
  - `f1rl/learning/evaluation/evaluator.py`
- Do not modify `f1rl/sandbox/**`, `f1rl/runtime/**`, `f1rl/learning/gym_environment.py`, observation/action/reward/race implementations, PPO implementation, training runner/vector env, track assets, or canonical experiment values.
- Do not run or inspect held-out-track performance in WP5.
- Canonical integration coverage may use training tracks only.
- No JSON/CSV persistence, provenance capture, statistical significance, confidence intervals, ranking, training orchestration, pilot training, or model selection.
- No production mocks/fakes/test doubles.
- Test policy doubles must live under `tests/learning/evaluation/mocks/`.
- Direct production SB3 import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`.
- Direct production Torch imports remain zero.

## Execution Environment

The WP5 worktree currently has no local `.venv`. The previously verified WP4 interpreter is available at:

```powershell
..\wp4-evaluation-telemetry\.venv\Scripts\python.exe
```

For every plan command, resolve the interpreter without falling back to system/Miniconda Python:

```powershell
$python = if (Test-Path '.\.venv\Scripts\python.exe') {
    (Resolve-Path '.\.venv\Scripts\python.exe').Path
} else {
    (Resolve-Path '..\wp4-evaluation-telemetry\.venv\Scripts\python.exe').Path
}
```

Then run tests as:

```powershell
& $python -m pytest ...
```

Do not treat a missing system package under another Python installation as a repository failure.

## Review Focus

1. Metrics must be derived from WP4 evidence, never reconstructed from reward or geometry when direct evidence exists.
2. Path length must include reset -> first accepted post-step position and must never include rejected collision candidate motion.
3. Collision count is obstacle-event count, not unique obstacles or collision ticks.
4. Modifier exposure is count of affected ticks, not entity-ID cardinality.
5. Normalized route progress must respect ordered checkpoint state and must not grant credit on later route segments.
6. Timeout duration must never be reported as completion time.
7. Evaluator must always call policy prediction with `deterministic=True`.
8. Evaluator must close the direct environment on success and every failure path.
9. WP5 must not become a held-out experiment runner or provenance/statistics layer.
10. `PPOModel` must remain unchanged; evaluation owns only a structural prediction boundary.

---

### Task 1: Define immutable episode metric contracts and validation
**Files:**
- Create: `f1rl/learning/evaluation/__init__.py`
- Create: `f1rl/learning/evaluation/metrics.py`
- Create: `tests/learning/evaluation/__init__.py`
- Create: `tests/learning/evaluation/test_metrics_contract.py`

**Interfaces:**
- Produces immutable `EpisodeMetrics` and `TrackEvaluationSummary`.
- No environment rollout exists yet.

- [ ] **Step 1: Write failing `EpisodeMetrics` contract tests**

Pin the exact approved fields:

```python
EpisodeMetrics(
    track_id="track-a",
    success=True,
    truncated=False,
    steps=12,
    episode_time=1.2,
    completion_time=1.2,
    episode_return=3.5,
    reward_component_totals=(
        ("target_progress", 4.7),
        ("collision", 0.0),
        ("time", -1.2),
    ),
    collision_count=0,
    path_length=10.0,
    final_normalized_route_progress=1.0,
    mean_speed=8.0,
    max_speed=12.0,
    airborne_steps=0,
    boost_applied_steps=2,
    slow_zone_applied_steps=0,
)
```

Require:
- frozen + slots dataclass;
- non-blank `track_id`;
- booleans are real `bool`, not integer substitutes;
- `steps >= 1`;
- finite non-negative time/path/speed values;
- `collision_count`, airborne/modifier step counts are non-negative integers;
- `0.0 <= final_normalized_route_progress <= 1.0`;
- reward component names are non-blank and unique;
- reward component values and episode return are finite;
- `completion_time` is finite/non-negative or `None`;
- `success=True` requires `completion_time` and `truncated=False`;
- `success=False` requires `completion_time is None`.

Do not require `episode_return == sum(component totals)` in the DTO constructor; that consistency belongs to the evaluator evidence check, not passive data storage.

- [ ] **Step 2: Write failing `TrackEvaluationSummary` contract tests**

Pin its approved descriptive fields and basic validity:
- non-blank track ID;
- `episode_count >= 1`;
- `0 <= success_count <= episode_count`;
- `success_rate == success_count / episode_count` within tolerance;
- all mean/count-like fields finite and non-negative where semantically required;
- normalized-progress mean in `[0,1]`;
- completion-time mean may be `None` only when `success_count == 0`.

- [ ] **Step 3: Run contract tests and verify RED**

```powershell
& $python -m pytest tests/learning/evaluation/test_metrics_contract.py -q
```

Expected: collection/import failure because the evaluation package does not exist.

- [ ] **Step 4: Implement only the immutable contracts**

Create `metrics.py` with small private scalar validation helpers and the two frozen/slots dataclasses.

Keep result values serialization-friendly:
- strings;
- bools;
- ints;
- floats;
- tuples of string/float pairs.

Do not import Gymnasium, SB3, Torch, environment classes, or track assets in this task.

- [ ] **Step 5: Export only the approved contract names**

Create `evaluation/__init__.py` and initially export:
- `EpisodeMetrics`;
- `TrackEvaluationSummary`.

Do not auto-export the evaluation package from base `f1rl.learning.__init__` in WP5.

- [ ] **Step 6: Run Task 1 tests**

```powershell
& $python -m pytest tests/learning/evaluation/test_metrics_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add f1rl/learning/evaluation tests/learning/evaluation
git commit -m "feat(evaluation): define episode metric contracts"
```

---

### Task 2: Implement normalized ordered-route progress and per-track descriptive aggregation
**Files:**
- Modify: `f1rl/learning/evaluation/metrics.py`
- Modify: `f1rl/learning/evaluation/__init__.py`
- Create: `tests/learning/evaluation/test_route_progress.py`
- Create: `tests/learning/evaluation/test_track_summary.py`

**Interfaces:**
- Consumes: `RaceRoute`, reset accepted vehicle position, final accepted position, `next_checkpoint_index`, and `finished`.
- Produces: private/pure normalized progress calculation plus public `summarize_track(...)`.

- [ ] **Step 1: Write failing straight and zero-checkpoint progress tests**

Pin a simple route:
- reset at `(0,0,0)`;
- finish at `(0,0,10)`;
- no checkpoints.

Assert:
- final position z=0 -> 0.0;
- z=5 -> 0.5;
- z=10 with unfinished state -> 1.0 by clamped projection;
- z<0 clamps to 0.0;
- z>10 clamps to 1.0;
- `finished=True` returns exactly 1.0.

The result is an evaluation progress measure; reaching the finish projection geometrically does not imply `success=True`.

- [ ] **Step 2: Write failing ordered multi-checkpoint tests**

Use:
```text
spawn (0,0,0)
-> cp0 (0,0,10)
-> cp1 (10,0,10)
-> finish (10,0,20)
```

Pin:
- `next_checkpoint_index=0`: projection only on spawn->cp0;
- after cp0, `next_checkpoint_index=1`: completed first segment + projection cp0->cp1;
- after cp1, `next_checkpoint_index=2`: completed first two segments + projection cp1->finish;
- being spatially beside/near finish while still requiring cp0 cannot receive later-segment credit;
- lateral offset does not inflate progress;
- projections behind/ahead clamp to current segment boundaries.

- [ ] **Step 3: Write failing 3D/ramp and degenerate-route tests**

Pin a segment with vertical displacement and verify full 3D dot-product projection/Euclidean length.

Reject:
- zero total reference route length;
- invalid `next_checkpoint_index`;
- malformed position tuple;
- non-finite coordinate.

- [ ] **Step 4: Implement the pure route-progress helper**

Recommended private signature:

```python
def _normalized_route_progress(
    route: RaceRoute,
    *,
    initial_position: tuple[float, float, float],
    vehicle_position: tuple[float, float, float],
    next_checkpoint_index: int,
    finished: bool,
) -> float:
    ...
```

Build waypoints exactly as approved:
```text
initial -> ordered checkpoint centers -> finish
```

Use only the current required segment for partial credit. Clamp final output to `[0.0, 1.0]` against floating-point edge noise.

Do not use target-progress reward values.

- [ ] **Step 5: Write failing `summarize_track` tests**

For a non-empty tuple of `EpisodeMetrics` sharing one track, assert:
- `episode_count`;
- `success_count`;
- `success_rate`;
- arithmetic mean of episode-level normalized progress;
- arithmetic mean of episode returns;
- arithmetic mean of per-episode collision counts;
- arithmetic mean of per-episode path lengths;
- arithmetic mean of each episode's `mean_speed`;
- arithmetic means of airborne/boost/slow step counts;
- successful-episode-only arithmetic mean completion time.

Pin explicitly that summary `mean_speed` is the **mean of episode-level means**, not a step-weighted pooled speed.

If no episodes succeeded, `mean_completion_time_for_successes is None`.

- [ ] **Step 6: Add summary validation failures**

Reject:
- empty episode tuple;
- non-`EpisodeMetrics` entries;
- mixed `track_id` values.

Do not add confidence intervals, standard deviations, rankings, significance tests, or seed/provenance fields.

- [ ] **Step 7: Implement `summarize_track` and export it**

Keep aggregation pure and deterministic.

Public exports after Task 2:
- `EpisodeMetrics`;
- `TrackEvaluationSummary`;
- `summarize_track`.

The normalized-progress helper stays module-private unless implementation evidence proves a public API is necessary.

- [ ] **Step 8: Run Task 2 tests**

```powershell
& $python -m pytest tests/learning/evaluation/test_route_progress.py tests/learning/evaluation/test_track_summary.py tests/learning/evaluation/test_metrics_contract.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

```powershell
git add f1rl/learning/evaluation tests/learning/evaluation
git commit -m "feat(evaluation): calculate route and track metrics"
```

---

### Task 3: Implement deterministic single-episode evaluator
**Files:**
- Create: `f1rl/learning/evaluation/evaluator.py`
- Modify: `f1rl/learning/evaluation/__init__.py`
- Create: `tests/learning/evaluation/mocks/__init__.py`
- Create: `tests/learning/evaluation/mocks/policies.py`
- Create: `tests/learning/evaluation/test_evaluator.py`

**Interfaces:**
- Consumes: `track_id`, `F1RLEnvFactory`, structural `EvaluationPolicy`.
- Produces: one `EpisodeMetrics` from exactly one direct environment episode.

- [ ] **Step 1: Create dedicated test policy doubles**

In `tests/learning/evaluation/mocks/policies.py`, define small reusable test-only policies such as:
- `ConstantPolicy`: returns a fixed valid NumPy action and records every `deterministic` argument;
- `ScriptedPolicy`: returns a finite action sequence;
- `FailingPolicy`: raises a configured exception from `predict`.

No test double enters production code.

- [ ] **Step 2: Write failing policy/factory boundary tests**

Pin:
- blank `track_id` rejected;
- non-`F1RLEnvFactory` rejected;
- object without callable `predict` rejected;
- evaluation calls `predict(observation, deterministic=True)` on every step;
- action validation remains owned by the environment/model boundary.

Define `EvaluationPolicy` as a runtime-checkable structural `Protocol` with only the approved `predict` method.

Do not import `PPOModel` or SB3 from `evaluator.py`.

- [ ] **Step 3: Write failing rollout lifecycle tests**

Using a small real `F1RLEnvFactory`, assert:
- one fresh environment is created for one `evaluate_episode` call;
- reset occurs before policy prediction;
- rollout stops immediately after `terminated or truncated`;
- environment closes after successful return;
- environment closes when policy prediction raises;
- environment closes when environment step raises.

Use pytest monkeypatch only to observe existing lifecycle calls where needed; reusable fake policy behavior remains in the dedicated mock module.

- [ ] **Step 4: Write failing basic metric accumulation tests**

Use real Gym infos/telemetry and a short deterministic fixture episode to pin:
- path starts from reset telemetry position;
- every post-step accepted position adds one 3D segment;
- `episode_return` sums scalar step rewards;
- reward-component totals preserve first-step key order and sum every step;
- drifting component names/order fails explicitly;
- collision count sums `len(obstacle_hit_ids)`;
- mean/max speed use post-step samples only;
- airborne count increments when post-step grounded is false;
- boost/slow exposure increments at most once per class per tick via `bool(ids)`.

- [ ] **Step 5: Pin terminal/time consistency checks**

During rollout require:
- `info["tick"] == executed_step_count`;
- `info["elapsed_time"]` is finite and equals `executed_step_count * factory.simulation_config.fixed_dt` within tight floating-point tolerance;
- final `success = bool(info["finished"])`;
- success implies `terminated=True, truncated=False`;
- failure at terminal implies `terminated=False, truncated=True`;
- `completion_time = episode_time` only on success, else `None`.

Reject a non-terminal loop ending by any evaluator-owned ad-hoc limit; WP5 uses the environment episode contract as the only stopping authority.

- [ ] **Step 6: Pin WP4 info validation**

Before consuming metric fields, require the existing metric-relevant shape:
- top-level `tick`, `elapsed_time`, `next_checkpoint_index`, `finished`, `reward_components`;
- nested telemetry position/speed/grounded/obstacle IDs/boost IDs/slow IDs.

Reject malformed/non-finite metric inputs explicitly.

Do not duplicate all WP4 schema validation for fields WP5 does not consume.

- [ ] **Step 7: Implement minimal evaluator accumulation**

Recommended public signature:

```python
def evaluate_episode(
    *,
    track_id: str,
    factory: F1RLEnvFactory,
    policy: EvaluationPolicy,
) -> EpisodeMetrics:
    ...
```

Implementation flow:
1. validate static arguments;
2. `env = factory.create()`;
3. in `try/finally`, reset and capture initial accepted position;
4. build `RaceRoute.from_track(factory.track)`;
5. deterministic predict -> direct `env.step`;
6. accumulate evidence;
7. at terminal, compute normalized route progress using final race info and accepted final position;
8. construct `EpisodeMetrics`;
9. always `env.close()` in `finally`.

No vector env, recorder object, evaluator class hierarchy, callback system, or persistence layer.

- [ ] **Step 8: Verify reward consistency**

Before returning metrics, require within floating-point tolerance:

```text
episode_return == sum(reward_component_totals)
```

This is a consistency assertion over environment evidence, not reward recomputation.

- [ ] **Step 9: Export evaluator API**

Public exports:
- `EvaluationPolicy`;
- `evaluate_episode`;
- existing metrics exports.

Do not modify base `f1rl.learning.__init__`.

- [ ] **Step 10: Run Task 3 tests**

```powershell
& $python -m pytest tests/learning/evaluation/test_evaluator.py tests/learning/evaluation/test_route_progress.py tests/learning/evaluation/test_metrics_contract.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit Task 3**

```powershell
git add f1rl/learning/evaluation tests/learning/evaluation
git commit -m "feat(evaluation): run deterministic episode evaluation"
```

---

### Task 4: Prove canonical training-track integration and PPO structural compatibility
**Files:**
- Create: `tests/integration/test_evaluator_metrics.py`
- Modify only test helpers if required.
- No production change expected unless integration exposes a real WP5 defect.

**Interfaces:**
- Consumes: checked-in WP1B experiment values, canonical **training** track assets, `F1RLEnvFactory`, WP4 telemetry, `EvaluationPolicy`, and existing `PPOModel`.
- Produces: end-to-end proof that WP5 measures the real stack without held-out inspection.

- [ ] **Step 1: Add a test-only canonical factory builder**

Inside the integration test, build the same real non-track contract from `ExperimentDefinition`:
- `VehicleParameters`;
- `SimulationConfig`;
- `SensorSuite` / `ObservationEncoder`;
- `RewardComposer`;
- `EpisodeConfig`;
- canonical initial state from the selected training asset.

Do not add a production builder to WP5.

- [ ] **Step 2: Evaluate `straight-sprint-v1` with a deterministic scripted/constant policy**

Use only the canonical training track.

Use the constant action:
```text
steering = 0
throttle = 1
```

A pre-implementation probe against the frozen current stack established the exact training-track fixture behavior:
```text
straight-sprint-v1
steps      = 95
elapsed    = 9.5 s
terminated = True
truncated  = False
finished   = True
```

Pin this as an integration regression, not as a scientific model-performance claim.

Assert reproducibility across two fresh evaluations with the same deterministic test policy:
- equal `EpisodeMetrics`;
- `success is True`;
- `truncated is False`;
- `steps == 95`;
- `episode_time == completion_time == 9.5` within floating-point tolerance;
- `final_normalized_route_progress == 1.0`;
- identical path length/return/progress diagnostics;
- all values finite and contract-valid.

Do not pin the exploratory probe's full floating-point episode return as a protocol constant; derive and validate the evaluator return from the actual rollout evidence.

Do not inspect any held-out asset.

- [ ] **Step 3: Prove WP4 event/modifier evidence feeds WP5 unchanged**

Use a canonical training setup that actually exercises at least one WP4 evidence class, preferably:
- `slalom-v1` for boost and/or obstacle evidence;
- `right-turn-v1` for slow-zone exposure.

The assertion is about metric derivation, not policy quality:
- event IDs produce the exact collision-event count;
- non-empty applied modifier IDs produce tick exposure counts;
- accepted positions produce path length.

If a simple scripted policy cannot reliably trigger a specific canonical mechanic, use a focused real `F1RLEnvFactory` fixture in this integration file rather than changing canonical assets.

- [ ] **Step 4: Prove `PPOModel` satisfies `EvaluationPolicy` structurally**

Construct a minimal real PPO model using an existing tiny valid factory/config fixture, without training it.

Assert:
- `isinstance(model, EvaluationPolicy)` when using the runtime-checkable Protocol;
- evaluator source has no SB3 import;
- calling `evaluate_episode` with `PPOModel` reaches the normal deterministic prediction boundary on a short environment contract.

Do not claim model performance or run held-out evaluation.

- [ ] **Step 5: Run WP5 integration tests**

```powershell
& $python -m pytest tests/integration/test_evaluator_metrics.py tests/integration/test_evaluation_telemetry.py -q
```

Expected: PASS.

- [ ] **Step 6: Run all WP5 focused tests**

```powershell
& $python -m pytest tests/learning/evaluation tests/integration/test_evaluator_metrics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

If only tests were needed:

```powershell
git add tests/integration/test_evaluator_metrics.py
git commit -m "test(evaluation): verify evaluator integration"
```

If integration exposes a real production defect, add a focused regression test first and include only the smallest demonstrated repair in this task commit.

---

### Task 5: Whole-WP5 scientific-boundary audit and exact verification
**Files:**
- Modify only if review exposes a real Critical/Important WP5 defect.
- No feature expansion.

**Interfaces:**
- Consumes: all WP5 commits from Tasks 1–4.
- Produces: exact verification evidence and a clean branch ready for local integration decision.

- [ ] **Step 1: Verify production-surface scope**

From base `032194d470b914c9e0d10fda8c19d5cfe6820630`, require production Python changes only under:

```text
f1rl/learning/evaluation/
```

No diff is permitted under:
- `f1rl/sandbox/**`;
- `f1rl/runtime/**`;
- `f1rl/learning/gym_environment.py`;
- `f1rl/learning/observation.py`;
- `f1rl/learning/action.py`;
- `f1rl/learning/rewards/**`;
- `f1rl/learning/episode/**`;
- `f1rl/learning/algorithms/**`;
- `f1rl/learning/training/**`;
- `assets/tracks/**`;
- `configs/experiments/ppo-generalization-v1.yml`.

- [ ] **Step 2: Audit held-out non-use**

Review all WP5 tests/source and require:
- no loading of `left-turn-v1`;
- no loading of `offset-slalom-v1`;
- no iteration over `split.heldout`;
- no held-out performance number captured in docs/tests;
- no protocol tuning based on evaluation outcome.

WP5 may know that a held-out partition exists, but must not execute it.

- [ ] **Step 3: Audit metric definitions against approved spec**

Check exact semantics:
1. collision = obstacle-event cardinality;
2. path = accepted 3D position distance including reset->first step;
3. mean/max speed = post-step samples;
4. airborne = post-step ungrounded tick count;
5. boost/slow = affected-tick counts;
6. success/time = finish vs timeout distinction;
7. route progress = ordered current-segment projection;
8. summary = descriptive episode-level means only;
9. same deterministic model/track reruns are not claimed as statistical replicates.

Any Critical/Important mismatch gets a regression test before repair.

- [ ] **Step 4: Run focused WP5 + WP4 regression**

```powershell
& $python -m pytest tests/learning/evaluation tests/integration/test_evaluator_metrics.py tests/integration/test_evaluation_telemetry.py tests/learning/test_gym_environment.py -q
```

Expected: PASS.

- [ ] **Step 5: Run learning/orchestration regressions**

```powershell
& $python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py tests/integration/test_vectorized_ppo_integration.py -q
```

Expected: PASS.

- [ ] **Step 6: Run full/static/dependency gate**

```powershell
& $python -m pytest -q
& $python -m compileall -q f1rl scripts
& $python -m pip check
git diff --check 032194d470b914c9e0d10fda8c19d5cfe6820630...HEAD
```

Audit production source and require:
- SB3 import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`;
- direct production Torch imports remain zero;
- production Mock/Fake markers remain zero;
- `f1rl/learning/evaluation/**` contains no SB3/Torch import;
- base `f1rl.learning.__init__` is unchanged;
- evaluation package does not import training runner/vector env;
- sandbox/runtime still do not import evaluation/learning back upward.

- [ ] **Step 7: Whole-WP5 review**

Review `032194d470b914c9e0d10fda8c19d5cfe6820630...HEAD` with priority on:
1. hidden held-out execution or tuning;
2. evaluator using stochastic predictions;
3. metric leakage into policy/environment;
4. wrong route-progress segment after ordered checkpoints;
5. timeout reported as completion;
6. collision/modifier counts using wrong cardinality semantics;
7. environment not closed on failure;
8. evaluator importing SB3 directly;
9. accidental persistence/provenance/statistics scope creep;
10. protocol/physics/training drift.

- [ ] **Step 8: Exact-final rerun**

After any review fix, rerun Steps 4–6 on the exact final HEAD.

Record:
- final HEAD;
- commit list;
- focused and full test counts;
- any platform-supported skip reason;
- compileall result;
- pip check result;
- diff-check result;
- dependency-boundary result;
- held-out non-use result;
- final `git status`.

- [ ] **Step 9: Final branch state and stop**

Require:

```text
branch == research/evaluator-metrics
git status --porcelain == empty
all verification commands green
no merge/push
WP6 not started
no held-out performance inspected
```

Stop at the WP5 integration gate and ask the user whether to merge, push, or keep the branch/worktree.

## Stop Condition

After Task 5, **STOP**. Do not begin WP6 Provenance automatically.
