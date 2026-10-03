# Evaluator + Metrics Design

**Status:** Written specification for review
**Branch:** `research/evaluator-metrics`
**Base HEAD:** `032194d470b914c9e0d10fda8c19d5cfe6820630`

## 1. Goal

WP5 turns WP4 evaluation telemetry into deterministic, reproducible episode metrics for the canonical research question:

> **How well does a PPO waypoint-guided navigation/control policy generalize to held-out track geometries?**

WP5 builds the **measurement layer only**. It does not train PPO, select final seeds, run the held-out scientific experiment, collect provenance, or tune protocol v1.

The evaluator must:
- run one deterministic policy episode on one explicitly supplied track/environment factory;
- consume only normal Gym outputs plus WP4 `info["telemetry"]`;
- calculate objective episode-level metrics with pinned aggregation rules;
- expose a small immutable result contract that WP8 can later compare across independently trained model seeds and tracks.

## 2. Why WP5 exists after WP4

WP4 deliberately exposed raw evidence without deciding how to aggregate it.

Available evidence now includes:

```text
top-level:
  tick
  elapsed_time
  next_checkpoint_index
  finished
  completed_checkpoint_id
  finished_now
  reward_components

telemetry:
  vehicle_position
  vehicle_velocity
  vehicle_speed
  grounded
  obstacle_hit_ids
  applied_boost_pad_ids
  applied_slow_zone_ids
```

WP5 owns the meaning of metrics derived from these fields.

It must not infer physics events that WP4 already exposes directly.

## 3. Selected architecture

Create a focused package:

```text
f1rl/learning/evaluation/
  __init__.py
  metrics.py
  evaluator.py
```

Expected responsibilities:

```text
metrics.py
  immutable episode metric DTOs
  pure route-progress calculation
  pure track-summary aggregation

evaluator.py
  minimal evaluation-policy protocol
  deterministic single-episode rollout
  step-by-step metric accumulation
```

No evaluator code belongs in:
- `sandbox`;
- `runtime`;
- PPO training/orchestration;
- track asset loading;
- the web/demo layer.

## 4. Evaluation execution model

WP5 evaluates **one direct `F1RLEnv` at a time**, created from an existing `F1RLEnvFactory`.

Conceptually:

```python
metrics = evaluate_episode(
    track_id="slalom-v1",
    factory=factory,
    policy=model,
)
```

The evaluator:
1. creates a fresh environment from the factory;
2. calls `reset()`;
3. repeatedly obtains an action from the policy with `deterministic=True`;
4. steps the environment;
5. accumulates metrics from reward + `info`;
6. stops only on `terminated or truncated`;
7. closes the environment in `finally`;
8. returns immutable `EpisodeMetrics`.

WP5 does not use `DummyVecEnv` or `SubprocVecEnv` for evaluation. Sequential direct evaluation avoids auto-reset ambiguity and keeps terminal metrics explicit.

## 5. Policy boundary

The evaluator must not directly import Stable-Baselines3.

Define the smallest structural protocol needed by evaluation:

```python
class EvaluationPolicy(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray: ...
```

Existing `PPOModel` satisfies this interface without modification.

The evaluator always calls:

```python
policy.predict(observation, deterministic=True)
```

There is no stochastic-evaluation option in protocol v1 because the canonical experiment already requires deterministic policy evaluation.

Test policy doubles, when needed, live only under dedicated test helper/mock files. They are never placed in production modules.

## 6. EpisodeMetrics contract

WP5 introduces one immutable episode result containing at least:

```python
EpisodeMetrics(
    track_id: str,
    success: bool,
    truncated: bool,
    steps: int,
    episode_time: float,
    completion_time: float | None,
    episode_return: float,
    reward_component_totals: tuple[tuple[str, float], ...],
    collision_count: int,
    path_length: float,
    final_normalized_route_progress: float,
    mean_speed: float,
    max_speed: float,
    airborne_steps: int,
    boost_applied_steps: int,
    slow_zone_applied_steps: int,
)
```

All values are plain immutable Python values. No `Vec3`, NumPy arrays, Gym spaces, environment objects, or SB3 objects are stored in the result.

## 7. Success, termination, and time semantics

For the current environment:

```text
terminated == valid race finish
truncated  == episode max_steps reached without finish
```

WP5 pins:

```python
success = bool(final_info["finished"])
```

The evaluator verifies consistency:
- successful episode => `terminated is True`, `truncated is False`;
- unsuccessful terminal episode => `terminated is False`, `truncated is True`.

Metrics:

```text
steps        = final info["tick"]
episode_time = final info["elapsed_time"]
```

`completion_time` is:

```python
episode_time if success else None
```

A timeout duration is therefore never mislabeled as a completion time.

## 8. Episode return and reward-component totals

`episode_return` is the arithmetic sum of the scalar reward returned by every environment step.

`reward_component_totals` independently sums each per-step value in:

```python
info["reward_components"]
```

The evaluator preserves the canonical component declaration/order observed on the first step and requires subsequent steps to expose the same component names in the same order.

For protocol v1 this is expected to remain:

```text
target_progress
collision
time
```

The evaluator verifies, within floating-point tolerance, that:

```text
episode_return == sum(all reward component totals)
```

It does not reconstruct reward from telemetry.

## 9. Collision count

WP4 already pins obstacle evidence to real `ObstacleHit` events.

WP5 defines:

```python
collision_count += len(info["telemetry"]["obstacle_hit_ids"])
```

Consequences:
- repeated hits against one obstacle on different ticks count repeatedly;
- two overlapping obstacle events in one tick count as two collisions/events;
- zero velocity or unchanged position does not count as a collision by itself.

This metric is an **obstacle-hit event count**, not a count of unique obstacles and not a count of collision-affected ticks.

## 10. Path length

Path length is computed from accepted vehicle positions only.

Let:
- `p0` = reset telemetry position;
- `pt` = accepted post-step telemetry position after step `t`.

Then:

```text
path_length = Σ distance(pt, p[t-1])
```

using 3D Euclidean distance.

A blocked collision that rolls the vehicle back to the previous accepted position contributes zero distance for that tick.

Rejected candidate penetration positions are never used.

## 11. Speed diagnostics

WP5 records accepted post-step speed once per executed step.

```text
mean_speed = arithmetic mean of post-step vehicle_speed samples
max_speed  = maximum post-step vehicle_speed
```

Reset speed is not included because reset is not a physics tick.

Every episode executes at least one step under the current positive `EpisodeConfig.max_steps`, so these diagnostics are always defined.

## 12. Airborne diagnostic

WP4 exposes post-step `grounded`.

WP5 defines:

```python
airborne_steps += int(not telemetry["grounded"])
```

This is deliberately a **discrete post-step occupancy count**, not an exact continuous airborne duration.

Because the canonical timestep is fixed, later reporting may derive:

```text
airborne_seconds = airborne_steps * fixed_dt
```

without pretending the telemetry identifies an exact within-tick takeoff/landing instant.

## 13. Boost and slow-zone exposure

WP4 exposes modifiers that actually influenced each physics tick through pre-step query semantics.

WP5 defines:

```python
boost_applied_steps += int(bool(applied_boost_pad_ids))
slow_zone_applied_steps += int(bool(applied_slow_zone_ids))
```

One tick counts at most once for each modifier class even if multiple same-class modifier IDs overlap.

This measures **ticks influenced by the mechanic**, not number of unique pads/zones and not summed entity-ID count.

With fixed `fixed_dt`, later reporting may convert these counts to exposure seconds.

## 14. Normalized route progress

A binary success metric is insufficient for failed/timeout episodes. WP5 therefore defines one continuous final progress metric in `[0, 1]`.

Build the route polyline from:

```text
initial vehicle position
-> checkpoint centers in RaceRoute order
-> finish center
```

Let route waypoints be:

```text
W0 = initial position
W1 = checkpoint 0 center
...
Wn = final checkpoint center
Wn+1 = finish center
```

The total reference route length is the sum of Euclidean segment lengths.

At the final state:
- if `finished == True`, normalized progress is exactly `1.0`;
- otherwise `next_checkpoint_index` identifies the current required segment;
- all earlier valid checkpoint segments count as complete;
- progress within the current segment is the clamped scalar projection of the accepted vehicle position onto that segment.

For current segment `A -> B` and vehicle position `P`:

```text
t = clamp(dot(P-A, B-A) / dot(B-A, B-A), 0, 1)

progress_distance =
    completed_reference_segment_length
    + t * length(A -> B)

normalized_route_progress =
    progress_distance / total_reference_route_length
```

This design has important properties:
- a vehicle cannot gain progress credit beyond the next required checkpoint merely by driving geometrically near a later checkpoint;
- lateral deviation does not directly inflate progress;
- the metric works with zero-checkpoint tracks as spawn -> finish;
- ramp vertical displacement is included because the projection is 3D;
- valid finish is always exactly 1.0.

A route with zero total reference length is invalid for WP5 evaluation and must fail explicitly rather than divide by zero.

## 15. Why route progress is not reward progress

`TargetProgressReward` measures per-step distance change to the fixed pre-step navigation target.

That reward is useful for learning but is not an adequate cross-track evaluation metric because:
- it accumulates local movement;
- it can be positive and negative repeatedly;
- total reward also contains collision/time penalties;
- different track layouts have different lengths.

WP5 normalized route progress is a separate evaluator metric based on ordered race state plus reference waypoint geometry.

WP5 does not change the reward function.

## 16. Per-track summary

WP3 requires held-out results to be reported per track and across independently trained model seeds.

WP5 may therefore provide a small pure aggregation function:

```python
summarize_track(episodes: tuple[EpisodeMetrics, ...]) -> TrackEvaluationSummary
```

The function requires all episodes to share one `track_id`.

The summary contains simple descriptive aggregates only:

```text
track_id
episode_count
success_count
success_rate
mean_final_normalized_route_progress
mean_episode_return
mean_collision_count
mean_path_length
mean_speed
mean_airborne_steps
mean_boost_applied_steps
mean_slow_zone_applied_steps
mean_completion_time_for_successes | None
```

No confidence interval, hypothesis test, significance claim, ranking, or generalization-gap conclusion belongs in WP5.

Those scientific comparisons belong to WP8 after independent trained-model replicates exist.

## 17. Independent-model replicate rule

A deterministic environment plus deterministic policy means rerunning the same model on the same track is not a new statistical replicate.

WP5 does not implement “N repeated episodes” as a source of statistical independence.

For later WP8 analysis, each episode result used as a replicate should come from an independently trained model seed.

WP5's summary function cannot prove model independence because seed/provenance identity belongs to WP6. It therefore only aggregates supplied episode metrics and documents this precondition.

## 18. Track partition boundary

WP5 must not automatically load or iterate the held-out partition.

The evaluator receives an explicit `track_id` + `F1RLEnvFactory`.

This prevents the metric layer from becoming an accidental experiment runner and keeps held-out access under later experiment orchestration.

During WP5 implementation/tests:
- synthetic fixtures are allowed;
- canonical training tracks may be used for integration coverage;
- held-out outcome inspection is unnecessary and should not be performed.

WP8 later owns actual training-vs-held-out evaluation.

## 19. Error handling

The evaluator fails explicitly when:
- `track_id` is blank;
- factory/policy contract is invalid;
- reset/step info lacks required WP4 fields;
- telemetry has malformed metric-relevant types;
- reward-component keys drift during an episode;
- final termination flags conflict with `finished`;
- `tick` or `elapsed_time` is inconsistent with the executed rollout;
- normalized route geometry has zero total length;
- policy prediction or environment step fails.

The evaluator does not fabricate partial `EpisodeMetrics` on failure.

The environment is still closed in `finally`.

## 20. Production boundaries

Expected new production surface:

```text
f1rl/learning/evaluation/__init__.py
f1rl/learning/evaluation/metrics.py
f1rl/learning/evaluation/evaluator.py
```

Possible minimal public exports:

```text
EpisodeMetrics
TrackEvaluationSummary
EvaluationPolicy
evaluate_episode
summarize_track
```

No direct Stable-Baselines3 or Torch imports are added to the evaluation package.

No changes are expected in:
- `f1rl/sandbox/**`;
- `f1rl/runtime/**`;
- `f1rl/learning/gym_environment.py`;
- observation/action/reward/race modules;
- training runner/vector env;
- track assets;
- canonical experiment environment values.

If implementation appears to require any of those changes, stop and reassess WP5 scope before proceeding.

## 21. Testing strategy

### Pure metric tests

Pin:
- 3D path-length accumulation;
- repeated/multi-obstacle event counting;
- speed mean/max;
- airborne-step counting;
- modifier-active step counting;
- reward total aggregation;
- optional completion time;
- exact normalized route progress for straight, multi-checkpoint, turn, ramp/3D, clamped-behind, and clamped-ahead cases;
- zero-reference-length rejection;
- same-track summary aggregation and mixed-track rejection.

### Evaluator contract tests

Use dedicated test policy helpers, not inline production mocks.

Pin:
- policy is always called with `deterministic=True`;
- fresh environment is created once per episode;
- rollout stops at terminal/truncated state;
- environment closes on success and on policy/env failure;
- returned metrics match actual step infos;
- malformed/drifting info fails explicitly.

### Integration tests

Use a canonical **training** track and the real `F1RLEnvFactory` contract.

Prove:
- WP4 telemetry feeds WP5 without adapter changes;
- one deterministic scripted policy rollout returns reproducible metrics;
- `PPOModel` is structurally compatible with `EvaluationPolicy` without evaluator SB3 imports;
- observation/action/reward behavior remains unchanged.

No held-out performance result is generated in WP5 tests.

## 22. Non-goals

WP5 does not:
- train or fine-tune PPO;
- choose PPO/sampling seeds;
- choose the final training budget/backend/vector count;
- run the WP7 pilot;
- run the WP8 baseline/generalization experiment;
- inspect held-out results for tuning;
- calculate confidence intervals or statistical significance;
- rank models;
- persist JSON/JSONL/CSV result artifacts;
- capture Git/dependency/platform provenance;
- modify telemetry;
- modify observation/action/reward/race semantics;
- modify track assets;
- add UI/demo visualization.

WP6 owns provenance. WP7 owns pilot training. WP8 owns the actual baseline/generalization study.

## 23. Acceptance criteria

WP5 is complete when:
1. a deterministic policy can be evaluated for one direct `F1RLEnvFactory` episode;
2. evaluator always uses `deterministic=True`;
3. episode metrics are immutable and serialization-friendly;
4. success/completion-time semantics distinguish finish from timeout;
5. return and reward-component totals are reproducibly accumulated;
6. collision count comes only from WP4 obstacle event IDs;
7. path length uses accepted positions including the reset->first-step segment;
8. speed, airborne, boost, and slow diagnostics use pinned per-step rules;
9. final normalized route progress is in `[0,1]` and respects ordered checkpoint state;
10. per-track summary rejects mixed-track episodes and computes descriptive means only;
11. no held-out experiment run or tuning occurs;
12. no SB3/Torch import boundary expands;
13. no sandbox/runtime/environment/protocol drift occurs;
14. focused and full test suites remain green.

## 24. Sequence after approval

```text
WP5 written spec
-> WP5 implementation plan
-> TDD pure metric primitives
-> TDD normalized route progress
-> TDD deterministic episode evaluator
-> TDD per-track summary
-> canonical training-track integration proof
-> whole-WP5 audit + exact verification
-> WP6 Provenance
```
