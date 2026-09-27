# Composable Transition Reward V1 Design

## Goal

Build a deterministic learning-layer reward system that scores one accepted simulation transition without owning simulation, episode, route-progress, or game-score state.

V1 rewards dense target progress, penalizes obstacle collision, and applies elapsed-time pressure. Checkpoint, finish-line, and score-pickup events are intentionally not rewarded yet because the current sandbox does not own validated race progress or terminal episode semantics.

## Architectural choice

```text
previous accepted vehicle
next accepted vehicle
active target before step
world events + dt
        |
        v
 RewardTransition
        |
        v
  RewardComposer
        |
        v
   RewardResult
```

The same navigation target must be used for both sides of one transition. Target selection remains external.

## Scope

### In scope

- Immutable `RewardTransition` input DTO.
- Immutable reward contribution/result DTOs.
- Composable ordered reward terms.
- Dense target-distance progress reward.
- Per-step collision penalty.
- `dt`-scaled time penalty.
- Deterministic contribution breakdown for debugging and ablation.
- Pure-Python implementation with sandbox dependencies only.

### Out of scope

- Checkpoint or finish bonuses.
- Score-pickup-to-RL-reward mapping.
- Episode termination/truncation.
- Next-checkpoint/lap progress ownership.
- Reward clipping, normalization, discounting, GAE, PPO, Gymnasium, NumPy, runtime ownership, or training orchestration.
- Stateful reward history, loop detectors, or moving averages.

## Package layout

```text
f1rl/learning/rewards/__init__.py
f1rl/learning/rewards/core.py
f1rl/learning/rewards/terms.py
f1rl/learning/__init__.py
f1rl/learning/README.md
tests/learning/rewards/test_reward_contract.py
tests/learning/rewards/test_reward_terms.py
tests/learning/rewards/test_reward_composer.py
tests/integration/test_reward_integration.py
```

No sandbox or runtime production file changes are required.

## Transition contract

```python
@dataclass(frozen=True, slots=True)
class RewardTransition:
    previous_vehicle: VehicleState
    next_vehicle: VehicleState
    target_position: Vec3 | None
    events: tuple[WorldEvent, ...]
    dt: float
```

Validation rules:

- `previous_vehicle` and `next_vehicle` must be `VehicleState` instances.
- `target_position` must be `Vec3` or `None`.
- `events` must be a tuple containing only current sandbox `WorldEvent` variants.
- `dt` must be numeric, finite, and strictly positive; booleans are rejected.
- construction is immutable.

The transition represents one already accepted world step. Reward code must not step physics or modify either state.

### Target timing invariant

`target_position` is the navigation target that was active **before** the step began. Both distance measurements use that same position:

```text
before distance = |target - previous_vehicle.position|
after distance  = |target - next_vehicle.position|
```

A route-progress layer may change targets only after this transition has been rewarded. This prevents artificial reward discontinuities when crossing a checkpoint.

## Result and contribution contract

```python
@dataclass(frozen=True, slots=True)
class RewardContribution:
    name: str
    value: float

@dataclass(frozen=True, slots=True)
class RewardResult:
    total: float
    contributions: tuple[RewardContribution, ...]
```

`RewardContribution.name` must be a non-empty string and `value` must be finite. `RewardResult.total` must be finite and `contributions` must be a tuple.

The composer constructs `total` as the ordinary ordered sum of contribution values. It performs no clipping or normalization.

Contribution order is exactly term declaration order. Names are stable analytics identifiers rather than display labels.

Built-in V1 names are exactly:

```text
target_progress
collision
time
```

## Reward term protocol

The learning reward package exposes a small standard-library protocol:

```python
class RewardTerm(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, transition: RewardTransition) -> RewardContribution: ...
```

Terms are stateless evaluators. Equal term configuration plus an equal transition must return an equal contribution.

All numeric term configuration uses the same validation rule:

- numeric and finite;
- `>= 0.0`;
- booleans rejected;
- `0.0` is valid and disables that term without changing composer structure.

A term must always return a finite contribution.

## Target progress term

```python
@dataclass(frozen=True, slots=True)
class TargetProgressReward:
    weight_per_meter: float
```

Its contribution name is `"target_progress"`.

When `target_position is None`, contribution is exactly `0.0`.

Otherwise:

```text
progress_meters = previous_distance - next_distance
value = weight_per_meter * progress_meters
```

Distances are ordinary 3D Euclidean distances. Positive progress gives positive reward; moving away gives negative reward; equal distance gives zero.

No clipping, dead zone, tolerance band, checkpoint interpretation, or target mutation occurs inside the term.

## Collision penalty term

```python
@dataclass(frozen=True, slots=True)
class CollisionPenalty:
    penalty: float
```

Its contribution name is `"collision"`.

If the transition contains one or more `ObstacleHit` events, contribution is exactly `-penalty`. Otherwise it is `0.0`.

Multiple obstacle-hit events in one step do **not** multiply the penalty. Overlapping authored obstacle volumes must not silently amplify RL semantics.

The term ignores checkpoint, finish-line, and score-pickup events.

## Time penalty term

```python
@dataclass(frozen=True, slots=True)
class TimePenalty:
    penalty_per_second: float
```

Its contribution name is `"time"` and value is exactly:

```text
-penalty_per_second * transition.dt
```

Scaling by elapsed seconds instead of ticks keeps the per-second pressure stable when fixed timestep changes.

For example:

```text
penalty_per_second = 0.1
dt = 0.1
contribution = -0.01
```

The term does not inspect speed, target, events, or score.

## Reward composer

```python
@dataclass(frozen=True, slots=True)
class RewardComposer:
    terms: tuple[RewardTerm, ...]

    def evaluate(self, transition: RewardTransition) -> RewardResult: ...
```

`terms` must be a tuple. Empty tuple is valid and yields `RewardResult(0.0, ())`.

Composer validation and evaluation rules:

- declaration order is preserved exactly; terms are never sorted;
- every term must expose a non-empty string `name` and callable `evaluate(...)`;
- term names must be unique within one composer;
- duplicate names raise `ValueError` at composer construction;
- evaluation calls each term exactly once in declaration order;
- each returned contribution name must equal that term's declared name;
- non-finite contribution values are rejected rather than clipped;
- `RewardResult.total` is the ordered sum of all contribution values.

A typical V1 composer is:

```python
RewardComposer((
    TargetProgressReward(1.0),
    CollisionPenalty(5.0),
    TimePenalty(0.1),
))
```

The composer owns no mutable episode state and is safe to reuse across independent environments when its immutable configuration is shared.

## Game score remains separate

`ScoreState` and `ScorePickupCollected.points` are game-facing mechanics. V1 reward terms do not read either value.

A `ScorePickupCollected` event may appear inside `RewardTransition.events`, but every built-in V1 term ignores it. Collectible points therefore cannot leak into RL reward accidentally.

This preserves the project invariant:

```text
game score != RL reward
```

A future collectible-reward experiment must add an explicit named reward term rather than reusing game score implicitly.

## Checkpoint and finish safety

Current sandbox interaction state only remembers zones occupied in the previous accepted step. Leaving and re-entering a checkpoint or finish zone emits its entry event again.

Therefore V1 deliberately assigns no reward to `CheckpointEntered` or `FinishLineEntered`. Rewarding those events now would allow repeated leave/re-enter farming.

Checkpoint-completion and finish bonuses require a separate race-progress/episode authority that can distinguish valid first completion, ordered progression, lap state, and terminal finish semantics.

## Determinism and immutability

For equal immutable term configuration and equal `RewardTransition` values, repeated evaluation must return equal `RewardResult` values.

Reward evaluation must not mutate:

- either vehicle state;
- event tuples or sandbox state;
- target position;
- term configuration;
- composer configuration;
- simulation/runtime state.

No term or composer may read randomness, wall clock, environment variables, network state, process-global mutable state, or hidden history.

## Integration acceptance

Integration tests may use `Simulation` to create real before/after snapshots, but production reward modules may not import runtime.

The acceptance flow is:

```text
before snapshot
active target
    -> Simulation.step(control)
after snapshot + events
    -> RewardTransition
    -> RewardComposer.evaluate(...)
```

Acceptance must prove at least:

- a real forward step toward a fixed target yields positive progress contribution plus time penalty;
- moving away yields negative progress contribution;
- a blocked obstacle step yields one collision penalty even with multiple `ObstacleHit` events;
- checkpoint/finish/pickup events do not create implicit reward;
- identical independent transitions produce identical breakdowns.

## Public exports

`f1rl.learning.rewards` exports:

```text
RewardTransition
RewardContribution
RewardResult
RewardTerm
RewardComposer
TargetProgressReward
CollisionPenalty
TimePenalty
```

`f1rl.learning` re-exports these public reward APIs while preserving existing observation and action exports.

## Dependency boundary

```text
sandbox VehicleState / Vec3 / WorldEvent
                 ^
                 |
        learning.rewards
                 ^
                 |
     future Gym / PPO / evaluation
```

Production reward code may depend on sandbox domain/event types and Python standard library only.

It must not import or depend on:

- `f1rl.runtime`;
- `f1rl.api` or `f1rl.drivers`;
- `f1rl.learning.observation` or `f1rl.learning.action`;
- Gymnasium, NumPy, PyTorch, Stable-Baselines3/PPO;
- FastAPI/WebSocket;
- React/Three.js/rendering code.

Sandbox must not import `f1rl.learning`.

## Verification criteria

The milestone is complete when tests prove:

1. `RewardTransition` validates immutable vehicle/target/event/positive-`dt` inputs.
2. Contribution/result DTOs reject invalid names, non-finite values, and non-tuple contributions.
3. Built-in term configuration accepts finite non-negative values, including `0.0`, and rejects booleans/non-finite/negative values.
4. Target progress uses the same explicit target for both vehicle states and returns signed 3D distance progress.
5. Missing target yields exactly zero target-progress contribution.
6. Collision penalty fires once per transition regardless of obstacle-hit count and ignores non-obstacle events.
7. Time penalty scales exactly by seconds: `-penalty_per_second * dt`.
8. Composer preserves declared order, requires unique names, evaluates each term once, and sums without clipping.
9. Empty composer returns `RewardResult(0.0, ())`.
10. Checkpoint, finish-line, and score-pickup events create no implicit built-in reward.
11. Reward evaluation is deterministic and does not mutate transition, terms, composer, or sandbox state.
12. Real simulation transitions integrate into reward calculation without production runtime dependency.
13. Existing physics/world/runtime/perception/observation/action tests remain green.
14. Reward production files have no forbidden framework/layer dependencies and sandbox does not import learning.

## Next milestone boundary

This reward layer supplies the fourth pure learning/runtime primitive needed by a future environment adapter:

```text
ObservationEncoder + PolicyAction mapping + Simulation.step + RewardComposer
```

Before a full racing Gym environment can safely reward checkpoint completion or terminate on finish, the project still needs explicit route-progress/episode semantics. Those semantics remain outside this reward milestone and must not be inferred from re-enterable sandbox entry events.
