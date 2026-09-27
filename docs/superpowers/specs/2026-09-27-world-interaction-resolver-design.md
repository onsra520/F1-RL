# World Interaction Resolver Design

## Goal

Build the deterministic world-step layer that turns TrackWorld membership into actual simulation behavior: boost acceleration, slow-zone drag, blocking obstacle contacts, checkpoint/finish entry events, score-pickup collection, and game-score updates.

This layer bridges the existing Track World Model and 3D physics core without coupling either one to rendering, networking, Gymnasium, PPO, or RL reward shaping.

## Architectural choice

V1 uses a **world-step orchestrator** rather than placing behavior inside entities or modifying physics derivatives after the fact.

- Entities remain immutable world definitions.
- TrackWorld remains a query-only spatial model.
- Physics receives generic motion modifiers, never BoostPad/SlowZone objects.
- The world step owns ordering, state transitions, collision rollback, event emission, and score updates.

Directly mutating `MotionDerivatives` outside physics is rejected because it would duplicate ground-forward/drag math. Letting entities mutate vehicles is rejected because it couples map definitions to simulation state.

## Scope

### In scope

- Generic grounded-motion modifiers for boost and slow effects.
- Deterministic checkpoint/finish entry detection.
- Score-pickup collection and game-score aggregation.
- Reference-point obstacle blocking with deterministic rollback.
- One fixed-step orchestration function producing the next immutable world state.
- Stable event ordering and deterministic repeated simulation.

### Out of scope

- Swept collision, continuous collision detection, vehicle-volume collision, penetration depth, impulses, restitution, or wall sliding.
- Ordered checkpoint/lap validation, lap timing, race completion policy, or respawn logic.
- RL rewards or reward shaping.
- Runtime wall-clock scheduling, WebSocket delivery, rendering, or editor behavior.

## Package layout

```text
f1rl/sandbox/dynamics/modifiers.py
f1rl/sandbox/interactions/state.py
f1rl/sandbox/interactions/events.py
f1rl/sandbox/interactions/resolver.py
f1rl/sandbox/rules/scoring.py
f1rl/sandbox/world.py
```

## Generic motion modifiers

`f1rl/sandbox/dynamics/modifiers.py` owns physics-facing values only:

```python
MotionModifiers(
    longitudinal_acceleration_bonus: float = 0.0,
    rolling_drag_multiplier: float = 1.0,
)
```

Both values must be finite. Acceleration bonus must be non-negative; drag multiplier must be at least `1.0`.

`compute_motion(...)` accepts an optional `MotionModifiers` value. Ground dynamics applies:

```text
longitudinal acceleration += longitudinal_acceleration_bonus
effective rolling drag = base rolling drag * rolling_drag_multiplier
```

Air dynamics ignores these grounded-track modifiers. Existing callers that omit modifiers preserve current behavior exactly.

Multiple overlapping boost pads sum `acceleration_bonus`. Multiple overlapping slow zones multiply `drag_multiplier`. Track effects are active only when a supporting `SurfaceContact` exists for the current query.

## Interaction state and entry events

Checkpoint and finish semantics are edge-triggered rather than emitted every simulation frame.

`f1rl/sandbox/interactions/state.py` owns:

```python
InteractionState(
    active_checkpoint_ids: frozenset[str],
    active_finish_line_ids: frozenset[str],
)
```

`f1rl/sandbox/interactions/events.py` owns immutable events:

```python
CheckpointEntered(checkpoint_id: str, order: int)
FinishLineEntered(finish_line_id: str)
ObstacleHit(obstacle_id: str)
```

The resolver compares the accepted final `TrackQuery` against previous `InteractionState`. A checkpoint/finish emits an entry event only when its ID is currently present and was absent in the previous active-ID set. Leaving clears membership; later re-entry emits again.

V1 does not enforce checkpoint ordering or decide whether a finish event completes a race.

## Score pickup and game score

Existing `collect_score_pickups(...)` remains the source of pickup-consumption state and `ScorePickupCollected` events.

`f1rl/sandbox/rules/scoring.py` owns immutable game/demo score state:

```python
ScoreState(points: int = 0)
```

and:

```python
apply_score_events(
    score: ScoreState,
    events: tuple[ScorePickupCollected, ...],
) -> ScoreState
```

Only pickup events add points in this milestone. Checkpoints and finish lines do not implicitly change score.

Game score remains independent from RL reward. `sandbox.rules.scoring` must not import `f1rl.learning`; a future reward composer may observe the same events and assign experiment-specific reward separately.

## Obstacle blocking in V1

TrackWorld currently reports obstacle membership for a queried **reference point**, not a full vehicle collider. V1 therefore implements a deliberately simple blocking rule:

1. Physics produces a candidate next `VehicleState`.
2. TrackWorld queries the candidate position.
3. If candidate position is inside one or more obstacles, the candidate movement is rejected.
4. The accepted vehicle keeps its previous position and orientation, sets linear/angular velocity to zero, and preserves the previous `grounded` flag.
5. One `ObstacleHit` event is emitted for each overlapping obstacle, ordered by obstacle ID.
6. The world is re-queried at the accepted rollback position before trigger/pickup state is resolved.

This prevents collection or checkpoint crossing through a blocked candidate position.

This is intentionally not a high-fidelity collision model. Large `dt` values may tunnel through thin obstacles because V1 does not perform swept collision detection. Full vehicle-volume collision belongs to the later collision subsystem.

## World state and step result

`f1rl/sandbox/world.py` owns the aggregate immutable simulation state for this milestone:

```python
WorldState(
    vehicle: VehicleState,
    track: TrackState,
    interactions: InteractionState,
    score: ScoreState,
)
```

The world-step output is:

```python
WorldStepResult(
    state: WorldState,
    query: TrackQuery,
    modifiers: MotionModifiers,
    events: tuple[WorldEvent, ...],
)
```

where `WorldEvent` is the union of `ObstacleHit`, `CheckpointEntered`, `FinishLineEntered`, and existing `ScorePickupCollected`.

The returned `query` describes the **accepted final position**, not a rejected collision candidate.

## Fixed-step orchestration

`step_world(...)` has one authoritative order:

```python
step_world(
    track: TrackWorld,
    state: WorldState,
    control: VehicleControl,
    parameters: VehicleParameters,
    dt: float,
    max_surface_distance: float,
) -> WorldStepResult
```

1. Query TrackWorld at the current vehicle position.
2. Resolve boost/slow `MotionModifiers` from current membership and supporting contact.
3. Compute physics motion using current surface contact and modifiers.
4. Integrate a candidate vehicle state.
5. Query TrackWorld at the candidate position.
6. If candidate intersects obstacles, reject candidate movement using the V1 blocking rule and re-query the accepted rollback position.
7. Resolve checkpoint/finish entry transitions from the accepted final query.
8. Collect currently overlapping uncollected score pickups.
9. Apply emitted pickup events to `ScoreState`.
10. Return the new immutable WorldState, accepted final query, modifiers used for this step, and stable event tuple.

Effects from a zone entered during the current movement begin on the next fixed step. This one-tick boundary keeps continuous effects deterministic without swept trigger detection.

## Event ordering and determinism

Events are returned in stable category order:

```text
ObstacleHit* -> CheckpointEntered* -> FinishLineEntered* -> ScorePickupCollected*
```

Within each category, events are ordered lexically by object ID. Repeated identical inputs must produce exactly equal `WorldStepResult` values.

Resolver functions and world stepping never mutate TrackWorld, WorldState, TrackState, VehicleState, or entity definitions in place.

## Validation

- `dt` must be strictly positive; existing integrator validation remains authoritative.
- `max_surface_distance` must be finite and non-negative; TrackWorld validation remains authoritative.
- Interaction active-ID sets must be `frozenset[str]` with non-empty string IDs.
- `ScoreState.points` must be a non-negative integer and reject booleans.
- Motion modifiers must reject NaN/Inf and invalid ranges.

## Dependency boundary

```text
track/ + entities/ + vehicle.py
          |
          v
interactions/ + rules/scoring.py
          |
          v
dynamics/ generic modifiers
          |
          v
sandbox/world.py
          |
          v
runtime/ (later)
```

No interaction/rule/world module may import learning, API, web, React, Three.js, or a third-party physics engine.

## Verification criteria

The subsystem is ready for runtime/session work when tests prove:

1. A boost pad increases grounded longitudinal acceleration by its configured bonus.
2. A slow zone multiplies grounded rolling drag without changing base parameters.
3. Multiple overlapping boost/slow zones combine deterministically.
4. Airborne motion ignores boost/slow modifiers.
5. Checkpoint and finish events emit only on entry, not every frame, and emit again after leave/re-entry.
6. A score pickup disappears from world state after collection and increases game score exactly once.
7. Game score remains independent from RL reward code.
8. A candidate obstacle hit rejects movement, zeros velocity, emits stable hit events, and prevents trigger/pickup collection at the rejected position.
9. Accepted final query always matches accepted final vehicle position.
10. Zone effects entered this step begin on the next step, not retroactively.
11. Repeated identical world steps are exactly deterministic.
12. Existing physics and TrackWorld callers remain backward compatible where optional modifiers are omitted.
13. Full sandbox tests remain green and dependency boundaries remain intact.
