# Race Progress / Episode Semantics V1 Design

## Goal

Define the authoritative single-lap racing progress and episode semantics that sit between deterministic world events and a future Gymnasium environment.

This layer owns ordered checkpoint completion, authoritative navigation-target selection, valid finish completion, and deterministic terminated/truncated status. It must not own physics stepping, observation encoding, reward calculation, policy actions, or Gymnasium/PPO integration.

## Architectural choice

```text
TrackWorld
   |
   v
RaceRoute          immutable authored race definition
   +
RaceProgressState  immutable episode progress
   +
World events
   |
   v
advance_race_progress(...)
   |
   v
RaceProgressResult
   |
   +--> authoritative next target
   +--> valid checkpoint completion diagnostic
   +--> valid finish diagnostic
   |
   v
episode_status(...)
   |
   +--> terminated
   +--> truncated
```

The design is deliberately pure and transition-based. `Simulation` remains unaware of racing episode semantics.

## Scope

### In scope

- Single-lap route definition.
- Immutable route checkpoints and finish target.
- `RaceRoute.from_track(track)` conversion.
- Strict route validation.
- Immutable race-progress state.
- Ordered checkpoint advancement from existing sandbox events.
- Ignore stale, repeated, future, or otherwise invalid checkpoint events.
- Valid finish only after all checkpoints complete.
- Authoritative current/next navigation target.
- Deterministic completion diagnostics.
- Episode `terminated` / `truncated` semantics using a maximum-step limit.
- Pure-Python implementation with sandbox dependencies only.

### Out of scope

- Multi-lap racing.
- Reward bonuses for checkpoint or finish completion.
- Physics/runtime stepping.
- Reset orchestration.
- Respawn/recovery rules.
- Lap timing or leaderboard logic.
- Reverse-direction detection.
- Off-track termination.
- Gymnasium, NumPy, PyTorch, Stable-Baselines3/PPO.
- Observation/action/reward ownership.

## Package layout

```text
f1rl/learning/episode/
├── __init__.py
├── route.py
├── progress.py
└── status.py
```

Expected tests:

```text
tests/learning/episode/test_route.py
tests/learning/episode/test_progress.py
tests/learning/episode/test_status.py
tests/integration/test_episode_integration.py
```

No sandbox or runtime production file needs to change.

## Route value objects

```python
@dataclass(frozen=True, slots=True)
class RaceCheckpoint:
    id: str
    order: int
    target_position: Vec3

@dataclass(frozen=True, slots=True)
class RaceFinish:
    id: str
    target_position: Vec3
```

Validation:

- IDs must be non-empty strings.
- `order` must be a non-negative integer; booleans are rejected.
- `target_position` must be `Vec3`.
- Value objects are immutable.

## RaceRoute

```python
@dataclass(frozen=True, slots=True)
class RaceRoute:
    checkpoints: tuple[RaceCheckpoint, ...]
    finish: RaceFinish

    @classmethod
    def from_track(cls, track: TrackWorld) -> RaceRoute: ...

    def target_for(self, state: RaceProgressState) -> Vec3 | None: ...
```

### Route invariants

`checkpoints` must be a tuple and `finish` must be `RaceFinish`.

Checkpoint orders must be:

- unique;
- contiguous;
- exactly `0..N-1` after ordering.

`RaceRoute` canonicalizes checkpoints by numeric `order`, never lexical ID.

A zero-checkpoint route is valid.

Checkpoint IDs must be unique within the route. The finish ID must not collide with any checkpoint ID.

### `RaceRoute.from_track`

`from_track` consumes the current `TrackWorld` authored entities and produces an immutable route.

Rules:

- each `Checkpoint` maps to `RaceCheckpoint(checkpoint.id, checkpoint.order, checkpoint.volume.center)`;
- exactly one `FinishLine` must exist;
- the finish maps to `RaceFinish(finish.id, finish.volume.center)`;
- zero checkpoints are allowed;
- zero or multiple finish lines are invalid for a V1 race route;
- checkpoint ordering is validated independently of `TrackWorld`'s lexical entity ordering.

The authored oriented-volume center is the V1 navigation target. No new target metadata is introduced.

## Progress state

```python
@dataclass(frozen=True, slots=True)
class RaceProgressState:
    next_checkpoint_index: int
    finished: bool
```

Initial state:

```python
RaceProgressState(next_checkpoint_index=0, finished=False)
```

Validation:

- `next_checkpoint_index` must be a non-negative integer; booleans are rejected;
- `finished` must be a bool.

The state intentionally does not store completed checkpoint IDs, lap count, event history, or mutable caches. On a valid single-lap route, `next_checkpoint_index == k` already means checkpoints `0..k-1` were accepted in order.

## State / route compatibility

Operations that consume both `RaceRoute` and `RaceProgressState` must reject impossible combinations:

- `next_checkpoint_index > len(route.checkpoints)` is invalid;
- `finished=True` requires `next_checkpoint_index == len(route.checkpoints)`.

An unfinished state with `next_checkpoint_index == len(route.checkpoints)` is valid: all checkpoints are complete and the finish is the active target.

## Authoritative target

`RaceRoute.target_for(state)` is the source of truth for route navigation targets.

Rules:

```text
unfinished + checkpoints remain
    -> checkpoints[next_checkpoint_index].target_position

unfinished + all checkpoints complete
    -> finish.target_position

finished
    -> None
```

For a zero-checkpoint route, the first target is immediately the finish center.

The episode package returns plain `Vec3 | None`; it must not import or construct `NavigationTarget` from the observation layer.

## Progress result

```python
@dataclass(frozen=True, slots=True)
class RaceProgressResult:
    state: RaceProgressState
    target_position: Vec3 | None
    completed_checkpoint_id: str | None
    finished_now: bool
```

Semantics:

- `completed_checkpoint_id` reports at most one newly accepted checkpoint for the transition;
- `finished_now` is true only on the transition that changes progress from unfinished to finished;
- no-op/ignored events return `completed_checkpoint_id=None` and `finished_now=False`;
- `target_position` is always exactly `route.target_for(result.state)`.

The result therefore carries the authoritative post-transition target directly for the caller.

## Progress transition API

```python
def advance_race_progress(
    route: RaceRoute,
    state: RaceProgressState,
    events: tuple[WorldEvent, ...],
) -> RaceProgressResult:
    ...
```

Validation:

- `route` must be `RaceRoute`;
- `state` must be `RaceProgressState` and compatible with the route;
- `events` must be a tuple of current `WorldEvent` variants.

The function is pure and does not mutate the route, state, or event tuple.

## Checkpoint advancement semantics

When checkpoints remain, only the checkpoint at `state.next_checkpoint_index` is valid.

For the expected checkpoint, an event is accepted only when both match:

```text
CheckpointEntered.checkpoint_id == expected.id
CheckpointEntered.order         == expected.order
```

A valid event increments `next_checkpoint_index` by exactly one and reports that checkpoint ID.

### Wrong/stale/future checkpoints

All other checkpoint-entry events are ignored, including:

- a future checkpoint;
- an already completed checkpoint re-entered later;
- an event with matching ID but wrong order;
- an event with matching order but wrong ID.

Ignored events do not rewind, skip, terminate, or mutate progress.

### Maximum one checkpoint per transition

A single call to `advance_race_progress` may complete at most one checkpoint.

Even if the event tuple contains multiple matching/future checkpoint entries due to overlapping authored volumes or synthetic input, the route may advance only one checkpoint stage in that accepted physics transition.

This prevents skipping multiple route stages in one step.

## Re-entry exploit prevention

Sandbox entry events are intentionally re-enterable after leaving a trigger volume.

Example:

```text
expected CP0
enter CP0  -> accepted, expected becomes CP1
leave CP0
re-enter CP0 -> event exists again, but expected is CP1
```

The repeated CP0 event is ignored because progress is keyed to the authoritative expected checkpoint rather than raw entry-event frequency.

## Finish semantics

A `FinishLineEntered` event may finish the race only when:

1. all checkpoints are already complete for the effective state of this transition; and
2. `finish_line_id == route.finish.id`.

An early finish event is ignored.

A wrong finish-line ID is ignored.

Once `finished=True`, later events are no-ops and `finished_now=False`.

## Final checkpoint and finish in the same transition

The final expected checkpoint and configured finish may both produce entry events in one accepted world transition, for example when authored volumes overlap.

V1 supports this case.

Processing semantics:

1. accept at most the one currently expected checkpoint;
2. if that acceptance makes all checkpoints complete, a matching configured finish event in the same event tuple may immediately finish the race;
3. no additional checkpoint may be accepted in the same transition.

Thus the transition may return both:

```text
completed_checkpoint_id = <final checkpoint id>
finished_now = True
```

This rule also naturally allows a zero-checkpoint route to finish on its first matching finish event.

## Event ordering

The result must not depend on lexical entity IDs.

For route semantics, the function conceptually evaluates:

1. the currently expected checkpoint against the whole event tuple;
2. then configured finish eligibility against the whole event tuple.

This ensures the valid final-checkpoint-plus-finish case works even if the raw tuple order is not relied upon by a future caller.

Obstacle-hit and score-pickup events are ignored by route progress.

## Reward timing invariant

This subsystem must preserve the fixed-target reward contract already established by `RewardTransition`.

Future Gym composition must use this sequence:

```text
1. target_before = route.target_for(progress_before)
2. encode current observation using target_before
3. map policy action
4. Simulation.step(...)
5. calculate RewardTransition using target_before for both before/after target distances
6. process the resulting world events with advance_race_progress(...)
7. next target = progress_result.target_position
8. encode the next observation with the next target
```

Progress must not switch the reward target before the current transition is scored.

## Episode status

```python
@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    max_steps: int

@dataclass(frozen=True, slots=True)
class EpisodeStatus:
    terminated: bool
    truncated: bool


def episode_status(
    progress: RaceProgressState,
    step_count: int,
    config: EpisodeConfig,
) -> EpisodeStatus:
    ...
```

Validation:

- `max_steps` must be an integer greater than zero; booleans are rejected;
- `step_count` must be a non-negative integer; booleans are rejected;
- `progress` must be `RaceProgressState`;
- `config` must be `EpisodeConfig`.

## Terminated / truncated precedence

Rules:

```text
progress.finished == True
    -> terminated=True
    -> truncated=False

progress.finished == False and step_count >= max_steps
    -> terminated=False
    -> truncated=True

otherwise
    -> terminated=False
    -> truncated=False
```

A valid finish at exactly the maximum step therefore wins over timeout:

```text
terminated=True
truncated=False
```

Timeout does not mutate race progress.

The Gym adapter will own reset orchestration later.

## Reward boundary

This milestone validates checkpoint completion and finish completion but does not add reward bonuses.

The existing V1 reward composer remains unchanged:

- target progress;
- collision penalty;
- time penalty.

A future reward experiment may consume validated progress diagnostics such as `completed_checkpoint_id` or `finished_now`, but it must not revert to rewarding raw re-enterable sandbox events.

## Dependency boundary

```text
sandbox TrackWorld / checkpoint / finish / WorldEvent / Vec3
                            ^
                            |
                  learning.episode
                            ^
                            |
              future Gym environment
```

Production `f1rl.learning.episode` may depend only on Python stdlib and sandbox domain/event types.

It must not import or depend on:

- `f1rl.runtime`;
- `f1rl.api` or `f1rl.drivers`;
- `f1rl.learning.observation`;
- `f1rl.learning.action`;
- `f1rl.learning.rewards`;
- Gymnasium, NumPy, PyTorch, Stable-Baselines3/PPO;
- FastAPI/WebSocket;
- renderer/web code.

Sandbox must not import `f1rl.learning`.

## Determinism and immutability

For equal immutable route/state/event/config inputs, repeated calls must return equal values.

The subsystem must not read randomness, wall clock, environment variables, network state, process-global mutable state, or hidden history.

No function may mutate authored track entities, event tuples, progress state, route objects, or runtime state.

## Public exports

`f1rl.learning.episode` exports:

```text
RaceCheckpoint
RaceFinish
RaceRoute
RaceProgressState
RaceProgressResult
advance_race_progress
EpisodeConfig
EpisodeStatus
episode_status
```

`f1rl.learning` re-exports the same episode APIs while preserving all existing observation, action, and reward exports.

## Integration acceptance

Integration tests may use the current sandbox/world/runtime to obtain real events, but production episode modules may not import runtime.

Acceptance must prove at least:

- track checkpoints authored in lexical-ID order different from numeric order still produce the correct route sequence;
- zero-checkpoint tracks target and complete the finish directly;
- checkpoint progress advances only in expected order;
- stale/re-entered and future checkpoint events are ignored;
- one transition cannot skip more than one checkpoint;
- early finish is ignored;
- the final checkpoint plus finish can complete in one transition;
- post-transition target always matches `route.target_for(result.state)`;
- finished state has no target;
- finish at exactly `max_steps` yields terminated, not truncated;
- timeout before finish yields truncated only;
- repeated equal inputs are deterministic and non-mutating.

## Verification criteria

The milestone is complete when tests prove:

1. Route value objects validate IDs, orders, target types, and immutability.
2. `RaceRoute` canonicalizes checkpoint order and rejects duplicate/non-contiguous orders and ID collisions.
3. `RaceRoute.from_track` requires exactly one finish line and uses authored volume centers as targets.
4. Zero-checkpoint routes are valid and target the finish immediately.
5. Progress-state values validate and incompatible route/state combinations fail fast.
6. `target_for` returns expected checkpoint, then finish, then `None` after completion.
7. Only the currently expected checkpoint can advance progress.
8. A transition can complete at most one checkpoint.
9. Re-entry, stale, future, mismatched-ID/order, obstacle, pickup, and early-finish events cannot advance valid progress incorrectly.
10. Final checkpoint plus matching finish may complete in the same transition.
11. `RaceProgressResult.target_position` always equals `route.target_for(result.state)`.
12. Finished progress is terminal and later events are no-ops.
13. Episode configuration and step count validation are strict.
14. Finish completion has precedence over timeout at the same step.
15. Timeout without finish truncates but does not terminate or mutate progress.
16. Evaluation is deterministic and immutable.
17. Existing physics/world/runtime/perception/observation/action/reward tests remain green.
18. Episode production code has no forbidden framework/layer dependencies and sandbox does not import learning.

## Next milestone boundary

After this layer, the project has the pure primitives needed for the first complete RL environment adapter:

```text
Simulation.step
+ ObservationEncoder
+ PolicyAction mapping
+ RewardComposer
+ RaceRoute / RaceProgressState
+ episode_status
--------------------------------
= Gymnasium environment
```

The Gymnasium adapter should compose these existing primitives rather than move their semantics into a new monolithic environment class.

Multi-lap rollover is deliberately absent from V1 and should be designed separately only if the research/training scenario later requires it.
