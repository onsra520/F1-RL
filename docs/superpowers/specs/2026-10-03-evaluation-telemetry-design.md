# Evaluation Telemetry Design

**Status:** Written specification for review
**Branch:** `research/evaluation-telemetry`
**Base HEAD:** `ea8465311de4ae883b38d0cd9faaba43a5f02401`

## 1. Goal

WP4 exposes raw, evaluation-only evidence from `F1RLEnv` so WP5 can compute objective episode metrics without changing the PPO policy interface.

The project is a 3D racing/navigation RL environment with:
- vehicle position, velocity, steering dynamics, drag, gravity, and grounded/airborne state;
- physical obstacles that can block motion;
- boost pads that add longitudinal acceleration;
- slow zones that multiply rolling drag;
- checkpoints and finish-line race semantics;
- PPO steering/throttle control;
- a scientific train/held-out geometry-generalization protocol.

WP4 must make the important physical and interaction state inspectable to an evaluator while preserving the frozen WP1B observation/action/reward contract.

## 2. Selected architecture

Telemetry is nested inside the existing Gymnasium `info` mapping:

```python
info["telemetry"] = {
    "vehicle_position": (x, y, z),
    "vehicle_velocity": (vx, vy, vz),
    "vehicle_speed": speed,
    "grounded": grounded,
    "obstacle_hit_ids": (...),
    "applied_boost_pad_ids": (...),
    "applied_slow_zone_ids": (...),
}
```

WP4 does not introduce:
- a public telemetry dataclass;
- a recorder subsystem;
- a wrapper around `F1RLEnv`;
- metric aggregation;
- telemetry storage.

The Gym adapter already owns episode-facing `info`, so this is the smallest layer that can expose evaluation evidence without pushing evaluation concepts into `sandbox` or `runtime`.

## 3. Why telemetry is nested

Existing top-level `info` fields describe episode/race state:

```text
tick
elapsed_time
next_checkpoint_index
finished
completed_checkpoint_id
finished_now
game_score
reward_components
```

The new fields describe physical/evaluation evidence. Nesting them under `telemetry` keeps those concerns explicit and avoids flattening progressively more evaluator-only fields into the episode contract.

All existing top-level keys remain unchanged.

## 4. Exact telemetry schema

The telemetry mapping contains exactly these seven keys:

```python
{
    "vehicle_position": tuple[float, float, float],
    "vehicle_velocity": tuple[float, float, float],
    "vehicle_speed": float,
    "grounded": bool,
    "obstacle_hit_ids": tuple[str, ...],
    "applied_boost_pad_ids": tuple[str, ...],
    "applied_slow_zone_ids": tuple[str, ...],
}
```

Domain objects such as `Vec3`, `VehicleState`, `TrackQuery`, `ObstacleHit`, `BoostPad`, or `SlowZone` are not placed in `info`.

This keeps the contract inspectable and serialization-friendly.

## 5. Reset semantics

Every successful `reset()` returns telemetry.

Physical state fields come from the reset `SimulationSnapshot.world.vehicle`:

```python
"vehicle_position" = reset vehicle position
"vehicle_velocity" = reset vehicle velocity
"vehicle_speed"    = reset vehicle speed
"grounded"         = reset vehicle grounded flag
```

Event/modifier evidence is empty on reset:

```python
"obstacle_hit_ids"         = ()
"applied_boost_pad_ids"    = ()
"applied_slow_zone_ids"    = ()
```

This remains true even if the spawn point lies inside a boost/slow volume. Reset does not execute a motion tick, so no modifier has yet been applied.

## 6. Step physical-state semantics

For a successful `step(action)`, the four vehicle fields describe the accepted **post-step** `VehicleState` from `snapshot_after.world.vehicle`.

This matters for collisions.

The world step computes a candidate vehicle state, then the obstacle resolver may reject that candidate and return a stopped/rolled-back accepted state. Telemetry records the accepted state, not a candidate penetration state.

Therefore an obstacle-blocked tick may legitimately report:
- the same position as the previous tick;
- zero velocity;
- zero speed.

That behavior is useful for later path-length and collision analysis.

## 7. Obstacle-event semantics

`obstacle_hit_ids` is derived directly from `ObstacleHit` values in `SimulationSnapshot.events` for the current tick.

It is not inferred from:
- reward values;
- reward component names;
- zero velocity;
- unchanged position;
- obstacle query presence.

Each emitted `ObstacleHit` contributes one ID.

Repeated hits against the same obstacle across multiple ticks remain repeated events. If one tick produces hits for multiple overlapping obstacles, every event ID is preserved.

The tuple follows the event order produced by the world interaction resolver. The current resolver order is deterministic because obstacle events are generated from obstacles sorted by ID.

## 8. Applied boost/slow semantics

Boost and slow telemetry must describe modifiers that **actually influenced the current physics tick**, not merely zones containing the post-step position.

The existing world step does:

```text
current state
  -> current TrackQuery
  -> resolve_motion_modifiers(current query)
  -> compute motion
  -> integrate
  -> collision/interactions
  -> accepted post-step state
```

The `SimulationSnapshot` immediately before `Simulation.step()` already contains the exact query corresponding to that current state.

Therefore `F1RLEnv.step()` captures:

```python
query_before = self._simulation.snapshot.query
```

before stepping.

After a successful step:

```python
applied_boost_pad_ids = sorted IDs from query_before.boost_pads
applied_slow_zone_ids = sorted IDs from query_before.slow_zones
```

**only when `query_before.contact is not None`.**

If `query_before.contact is None`, both applied-ID tuples are empty because `resolve_motion_modifiers()` returns default modifiers while airborne/no-contact.

This exactly mirrors existing physics semantics.

## 9. Zone entry/exit timing

Because modifiers are resolved from the query at the start of a tick:

- entering a boost/slow zone during tick N does not mark that modifier as applied for tick N;
- if the accepted state ends inside the zone, it can be applied on tick N+1;
- leaving a zone during tick N still reports the modifier as applied for tick N if the vehicle began that tick inside the zone;
- being spatially inside a modifier volume without ground contact does not count as applied.

WP4 does not change this behavior.

## 10. Deterministic ID ordering

`applied_boost_pad_ids` and `applied_slow_zone_ids` are returned sorted lexicographically by ID.

This mirrors `resolve_motion_modifiers()`, which sorts boost pads and slow zones by ID before composing their numeric effects.

`obstacle_hit_ids` preserves event order, which is already deterministic under the current resolver.

## 11. No policy-information leak

WP4 must not modify:
- `ObservationConfig`;
- `ObservationEncoder`;
- observation feature names/order;
- observation normalization;
- observation-space shape;
- action-space shape/range;
- reward components or weights;
- race-progress semantics.

Protocol-v1 observations remain the same 16 floats on the canonical experiment:

```text
7 rangefinders
+ local velocity xyz
+ local target direction xyz
+ target distance
+ target_available
+ grounded
```

Absolute position, obstacle IDs, boost IDs, slow-zone IDs, and evaluation telemetry remain unavailable to the policy.

Data flow stays one-way:

```text
TrackWorld / vehicle physics
        |-----------------> ObservationEncoder -> PPO
        |
        +-----------------> telemetry -> evaluator
```

There is no `telemetry -> ObservationEncoder` path.

## 12. Existing info compatibility

The step result keeps the current top-level structure and adds only the nested telemetry mapping:

```python
{
    "tick": ...,
    "elapsed_time": ...,
    "next_checkpoint_index": ...,
    "finished": ...,
    "completed_checkpoint_id": ...,
    "finished_now": ...,
    "game_score": ...,
    "reward_components": {...},
    "telemetry": {...},
}
```

Reset returns the same top-level keys with its existing zero-step values plus reset telemetry.

No existing key is renamed or removed.

## 13. Multi-track wrapper semantics

`_SampledF1RLEnv` continues to add:

```python
"track_id": scenario.track_id
```

at the top level by copying the underlying `info`.

Telemetry therefore survives unchanged alongside `track_id` on reset and step.

WP4 does not modify track sampling or scenario selection.

## 14. VecEnv terminal semantics

Stable-Baselines3 `DummyVecEnv` automatically resets an environment slot after a terminal step.

The info returned for that terminal transition must retain:
- the terminal episode's `track_id` when using sampled multi-track environments;
- the terminal step's `telemetry`;
- the terminal observation inserted by `DummyVecEnv`.

The terminal info must not be overwritten by telemetry from the automatically reset next episode.

WP4 adds regression coverage for this behavior. No VecEnv implementation change is expected unless the test exposes a real compatibility defect.

## 15. Canonical mechanics coverage

The amended six-track corpus now gives telemetry real canonical coverage:

```text
slalom-v1          -> boost
offset-slalom-v1   -> boost
right-turn-v1      -> slow zone
left-turn-v1       -> slow zone
slalom pair        -> blockers
ramp-v1            -> elevation / grounded-airborne behavior
```

WP4 tests should include canonical integration evidence that:
- a tick beginning inside `boost-0` while grounded reports `("boost-0",)`;
- a tick beginning inside `slow-0` while grounded reports `("slow-0",)`;
- an airborne/no-contact state does not report a spatially overlapping modifier as applied;
- a real blocker hit reports the corresponding obstacle ID.

## 16. Relationship to WP5 metrics

WP4 provides raw evidence only.

WP5 may derive:

```text
success                  <- finished
completion_time          <- elapsed_time
steps                    <- tick
episode_return           <- sum of rewards
reward components        <- reward_components
collision_count          <- sum(len(obstacle_hit_ids))
path_length              <- sum(distance(position[t], position[t-1]))
speed diagnostics        <- vehicle_speed / vehicle_velocity
airborne duration        <- grounded
boost exposure           <- applied_boost_pad_ids
slow-zone exposure       <- applied_slow_zone_ids
normalized route progress <- race progress + track geometry + vehicle position
```

WP4 does not define aggregation rules or scientific metric priority.

## 17. Production ownership

Expected production change is limited to:

```text
f1rl/learning/gym_environment.py
```

The recommended implementation uses private helper(s) inside that module for:
- converting a `VehicleState` to scalar/tuple telemetry;
- extracting obstacle event IDs;
- extracting applied modifier IDs from the pre-step query.

No public API export is required.

No changes are expected in:
- `f1rl/sandbox/**`;
- `f1rl/runtime/**`;
- `f1rl/learning/observation.py`;
- reward modules;
- training runner;
- track assets;
- experiment YAML.

## 18. Error and lifecycle behavior

Telemetry construction happens only after the existing relevant state is valid.

Reset:
- if observation encoding fails, reset continues to fail exactly as today;
- no successful `info` is returned.

Step:
- invalid policy actions still fail before simulation mutation;
- simulation failures still do not fabricate telemetry;
- reward/progress/observation failures after simulation mutation still require reset under the existing lifecycle;
- telemetry must not change that failure behavior.

Telemetry helpers should not introduce independent mutable state.

## 19. Required tests

### Reset contract

Verify:
- exact seven telemetry keys;
- position/velocity are plain 3-tuples of floats;
- speed is a float;
- grounded is bool;
- event/modifier tuples are empty;
- repeated resets and different Gym seeds return identical telemetry for the deterministic initial state.

### Step physical telemetry

Verify:
- post-step position/velocity/speed/grounded match `SimulationSnapshot.world.vehicle`;
- `speed == magnitude(velocity)`;
- a blocked obstacle tick reports accepted rollback/stopped state plus `obstacle_hit_ids`;
- repeated obstacle hits remain repeatable across ticks when the physical setup generates them.

### Modifier telemetry

Verify with canonical assets:
- grounded start inside slalom `boost-0` -> applied boost ID present;
- grounded start inside right-turn `slow-0` -> applied slow ID present;
- entering a zone during a tick does not mark it applied until the next tick;
- leaving a zone during a tick still reports it for that tick when present at tick start;
- no-contact state -> applied boost/slow tuples empty.

### Observation/reward regression

Pin before/after equality for:
- deterministic reset observation;
- observation-space shape and bounds;
- feature count/order contract;
- action space;
- reward result for the same transition.

Telemetry addition must be the only semantic difference in the Gym return info.

### VecEnv regression

Verify:
- standard `DummyVecEnv` preserves nested telemetry;
- sampled wrapper preserves telemetry alongside `track_id`;
- terminal auto-reset retains terminal telemetry and terminal track ID.

## 20. Non-goals

WP4 does not:
- compute collision count;
- compute path length;
- compute normalized route progress;
- aggregate speed/airborne/modifier statistics;
- write JSON/JSONL/CSV evaluation artifacts;
- implement the evaluator;
- alter physics;
- alter track content;
- alter PPO observations;
- alter rewards;
- alter training configuration;
- choose seeds or training budget;
- add telemetry to sandbox/runtime domain contracts.

## 21. Acceptance criteria

WP4 is complete when:
1. every successful reset and step returns the exact seven-field nested telemetry schema;
2. physical telemetry reflects accepted vehicle state;
3. obstacle IDs come from actual `ObstacleHit` events;
4. boost/slow IDs reflect modifiers actually applied at tick start, including contact gating;
5. reset applied/event tuples are empty;
6. canonical boost, slow, obstacle, and grounded/no-contact behaviors are covered by integration tests;
7. observation/action/reward/race semantics remain unchanged;
8. sampled and vectorized environments preserve telemetry, including terminal transitions;
9. no sandbox/runtime/track/config protocol drift occurs;
10. full test suite and dependency boundaries remain green.

## 22. Sequence after approval

```text
WP4 written spec
-> WP4 implementation plan
-> TDD reset/step telemetry contract
-> TDD modifier/event semantics
-> VecEnv terminal preservation tests
-> canonical integration proof
-> whole-WP4 review + verification
-> WP5 Evaluator + Metrics
```
