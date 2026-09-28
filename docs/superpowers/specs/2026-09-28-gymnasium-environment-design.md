# Gymnasium Environment V1 Design

## Goal

Add the first standards-compliant RL environment adapter for F1-RL by composing the deterministic primitives already implemented in sandbox, runtime, and learning.

`F1RLEnv` is a thin Gymnasium adapter. It must not reimplement physics, observation features, action mapping, reward terms, route progress, or episode termination rules.

This milestone is the final integration layer before PPO/model work.

## Architectural position

```text
TrackWorld + initial WorldState
          |
          v
      Simulation
          |
          +--> ObservationEncoder
          +--> PolicyAction mapping
          +--> RewardComposer
          +--> RaceRoute / RaceProgressState
          +--> episode_status
          |
          v
       F1RLEnv
          |
          v
 future PPO / vectorized training
```
## Scope

### In scope

- `f1rl.learning.gym_environment.F1RLEnv`.
- Gymnasium `Env` compliance for `reset()` and `step()`.
- Fixed-track environment instances.
- NumPy `float32` action and observation spaces.
- Composition of existing action, observation, reward, route-progress, and episode-status primitives.
- Deterministic reset semantics.
- Primitive diagnostic `info` values.
- Gymnasium environment checker acceptance.
- Runtime dependencies on Gymnasium and NumPy.

### Out of scope

- PPO, Stable-Baselines3, PyTorch, policy networks, or training loops.
- Vectorized environments.
- Track randomization inside `reset()`.
- Curriculum learning or domain randomization.
- Multi-lap racing.
- Rendering (`human`, `rgb_array`, Three.js, pygame).
- Gym global registry / `gym.make()` registration.
- New reward terms or checkpoint/finish reward bonuses.
- New physics, action, observation, or route semantics.

## Package and dependency location

Create `f1rl/learning/gym_environment.py` and re-export its public API from `f1rl.learning`.
Update `pyproject.toml` runtime dependencies to include:

```toml
gymnasium>=1.3,<2
numpy>=2,<3
```

Do not add Stable-Baselines3 or PyTorch in this milestone.

Dependency direction:

```text
sandbox <- runtime
sandbox <- learning primitives
runtime + learning primitives <- gym_environment
```

`sandbox` and `runtime` must not import Gymnasium or NumPy because those lower layers remain reusable framework-free primitives.

## Environment construction

```python
class F1RLEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        track: TrackWorld,
        initial_state: WorldState,
        vehicle_parameters: VehicleParameters,
        simulation_config: SimulationConfig,
        observation_encoder: ObservationEncoder,
        reward_composer: RewardComposer,
        episode_config: EpisodeConfig,
    ) -> None:
        ...
```
Construction rules:

- Validate all constructor arguments with strict concrete-type checks consistent with existing project contracts.
- Construct one `Simulation` from the supplied track, initial state, parameters, and simulation config.
- Construct the authoritative route with `RaceRoute.from_track(track)`; callers do not supply a second independent route.
- Initialize progress as `RaceProgressState(0, False)` only during `reset()`, not during construction.
- Store the supplied observation encoder, reward composer, simulation config, and episode config.
- A single environment instance is permanently bound to one `TrackWorld` for its lifetime.
- Construction does not implicitly start an episode. `reset()` is required before `step()`.

The fixed-track rule deliberately keeps track sampling outside the environment. Future vectorized/model code will build multiple environments or factories with different tracks.

## Gymnasium metadata

V1 has no renderer:

```python
metadata = {"render_modes": []}
render_mode = None
```

Do not implement image capture, pygame, browser rendering, or a fake render mode.

## Action space

The existing policy action contract is authoritative:

```text
steering in [-1, 1]
throttle in [0, 1]
```
Expose:

```python
action_space = gymnasium.spaces.Box(
    low=np.array([-1.0, 0.0], dtype=np.float32),
    high=np.array([1.0, 1.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32,
)
```

`step(action)` requires a NumPy array with shape `(2,)` containing finite numeric values. Values are converted to Python floats and passed through `PolicyAction`, which remains the final bounds validator. Do not clamp or remap invalid actions.

Conversion path:

```text
np.ndarray action
    -> PolicyAction(steering, throttle)
    -> map_policy_action(...)
    -> VehicleControl
```

Action validation must happen before `Simulation.step()` so invalid actions cannot mutate simulation state.

## Observation space

The observation shape is exactly `(observation_encoder.size,)` and dtype is `np.float32`.

The existing feature layout is authoritative:

```text
N rangefinder distances       [0, 1]
3 local velocity values       [-1, 1]
3 local target-direction      [-1, 1]
target distance               [0, 1]
target available              [0, 1]
grounded                      [0, 1]
```
Construct `observation_space.low` and `.high` arrays from the declared sensor count plus the fixed feature layout above. Do not infer bounds from runtime samples.

Observation encoding path:

```text
ObservationEncoder.encode(...)
    -> tuple[float, ...]
    -> np.asarray(..., dtype=np.float32)
```

Every observation returned from `reset()` or `step()` must satisfy `observation_space.contains(observation)`.

## Navigation target adaptation

Race progress owns a plain `Vec3 | None`; the observation encoder accepts `NavigationTarget | None`.

The Gym adapter performs only this representation bridge:

```python
NavigationTarget(target_position) if target_position is not None else None
```

The Gym adapter must never choose, reorder, skip, or infer route targets itself.

## Reset contract

Signature follows Gymnasium:

```python
def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    ...
```

`reset()` must call `super().reset(seed=seed)`.
Reset semantics:

1. `options is None` or an empty mapping is accepted.
2. Any non-empty options mapping is rejected in V1 rather than silently ignored.
3. `Simulation.reset()` restores the exact constructor-supplied initial `WorldState`.
4. Race progress becomes `RaceProgressState(0, False)`.
5. The initial authoritative target comes from `route.target_for(progress)`.
6. The initial observation is encoded from the reset vehicle and initial target.
7. Lifecycle state becomes active and `step()` is allowed.
8. Reset info contains zero-step diagnostics and no reward contributions.

V1 contains no stochastic environment behavior. A supplied Gymnasium seed is accepted and initialized through the base class but does not alter track, physics, spawn state, or route.

## Step contract

Signature follows Gymnasium:

```python
def step(
    self,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
    ...
```

A step is legal only while an episode is active.

The exact composition order is authoritative:

```text
1. validate action
2. target_before = route.target_for(progress_before)
3. previous_vehicle = simulation.snapshot.world.vehicle
4. map action -> VehicleControl
5. snapshot_after = simulation.step(control)
6. evaluate RewardTransition using target_before
7. advance_race_progress using snapshot_after.events
8. compute episode_status using snapshot_after.tick
9. encode next observation using post-progress target
10. construct info
11. if terminated or truncated, require reset before another step
```
## Reward transition

The adapter constructs:

```python
RewardTransition(
    previous_vehicle=previous_vehicle,
    next_vehicle=snapshot_after.world.vehicle,
    target_position=target_before,
    events=snapshot_after.events,
    dt=simulation_config.fixed_dt,
)
```

The returned Gym reward is exactly `float(reward_result.total)`.

Do not add clipping, normalization, checkpoint bonuses, finish bonuses, score conversion, or other Gym-only reward behavior.

Game score remains independent from RL reward.

## Progress and episode status

After reward evaluation, call:

```python
progress_result = advance_race_progress(route, progress_before, snapshot_after.events)
status = episode_status(progress_result.state, snapshot_after.tick, episode_config)
```

The environment stores `progress_result.state` as its next progress state.

Consequences inherited from the existing episode layer remain unchanged:

- ordered checkpoint progression;
- at most one checkpoint per transition;
- early finish ignored;
- final checkpoint plus finish may complete in one transition;
- valid finish yields `terminated=True`;
- unfinished step limit yields `truncated=True`;
- valid finish exactly at the limit yields terminated-only.
## Lifecycle and failure semantics

The environment has an explicit lifecycle gate:

```text
constructed -> reset required
reset       -> active episode
terminated  -> reset required
truncated   -> reset required
```

Calling `step()` before the first reset or after a terminal/truncated step raises `RuntimeError`.

Invalid actions are rejected before simulation mutation and leave the episode active and unchanged.

`Simulation.step()` failures preserve simulation state by the existing runtime contract; the Gym adapter must also leave race progress unchanged.

If an exception occurs after `Simulation.step()` has successfully advanced but before the Gym step result is completed (for example a custom reward term raises), the adapter marks the episode as requiring reset before re-raising. This prevents continuation with simulation and learning state out of sync.

A subsequent `reset()` always recovers by restoring the constructor-supplied initial simulation state and fresh race progress.

## Info contract

`info` contains diagnostics only and must remain serializable/practical for training logs. Do not expose the complete mutable object graph.

Return these keys on each step:

```python
{
    "tick": int,
    "elapsed_time": float,
    "next_checkpoint_index": int,
    "finished": bool,
    "completed_checkpoint_id": str | None,
    "finished_now": bool,
    "game_score": int,
    "reward_components": dict[str, float],
}
```
`reward_components` is built from `RewardResult.contributions` in declaration order. Reward term names are already unique by `RewardComposer` contract.

Reset info uses the same diagnostic shape where meaningful:

```python
{
    "tick": 0,
    "elapsed_time": 0.0,
    "next_checkpoint_index": 0,
    "finished": False,
    "completed_checkpoint_id": None,
    "finished_now": False,
    "game_score": initial_score,
    "reward_components": {},
}
```

No `WorldState`, `TrackQuery`, `SimulationSnapshot`, or event objects are placed in info in V1.

## Registration and factories

Do not call `gymnasium.register()` and do not define a global environment ID in V1.

Users and future training code construct `F1RLEnv(...)` directly. Future model/vectorized code may define factories such as `make_env(track_config)`, but factory and track-sampling policy are outside this milestone.

## Public API

`f1rl.learning` re-exports:

```text
F1RLEnv
```

All existing action, observation, reward, and episode exports remain unchanged.

No Gymnasium types are re-exported from sandbox or runtime.

## Determinism

For equal constructor inputs and equal action sequences, each reset episode produces equal observations, rewards, progress diagnostics, and terminal/truncation decisions.
Calling `reset(seed=...)` is reproducible but does not change deterministic physics because V1 does not consume RNG state.

## Testing and acceptance

Create focused Gym environment tests, expected under:

```text
tests/learning/test_gym_environment.py
```

Add integration coverage only where real world stepping is required.

Acceptance must prove:

1. Constructor derives `RaceRoute` from the supplied track and rejects invalid race tracks through the existing route contract.
2. `action_space` has exact V1 bounds, shape `(2,)`, and `float32` dtype.
3. `observation_space` has exact encoder-derived size, feature bounds, and `float32` dtype.
4. `reset()` returns `(np.ndarray, info)` and the observation is inside `observation_space`.
5. Repeated resets restore the exact initial simulation state and fresh race progress.
6. Empty options are accepted; non-empty options are rejected.
7. `step()` before reset is rejected.
8. `action_space.sample()` actions are accepted.
9. Wrong action type/shape/non-finite/out-of-range values fail before simulation mutation.
10. One legal step advances exactly one simulation tick.
11. Gym reward equals `RewardComposer` total with no added transformation.
12. Current reward uses the pre-step target.
13. Next observation uses the post-progress target.
14. Valid checkpoint progression changes the next target correctly.
15. Valid finish returns `terminated=True, truncated=False`.
16. Step-budget exhaustion before finish returns `terminated=False, truncated=True`.
17. Finish exactly at the step budget remains terminated-only.
18. `step()` after terminated/truncated requires reset.
19. `game_score` remains distinct from RL reward.
20. Reward breakdown appears in primitive `info` diagnostics.
21. Returned step observations are `float32` and contained by the declared observation space.
22. Reward/progress/observation composition is deterministic for equal inputs.
23. A failure after simulation advancement marks the environment reset-required before re-raising.
24. Reset recovers from that faulted state.
25. `gymnasium.utils.env_checker.check_env` accepts the environment with render checking skipped.
26. No Gym global registration is introduced.
27. No rendering dependency or render mode is introduced.
28. Existing sandbox/runtime/learning primitive tests remain green.
29. Sandbox and runtime production code do not import Gymnasium or NumPy.
30. Gym environment production code does not import Stable-Baselines3 or PyTorch.

The Stable-Baselines3 `check_env()` is intentionally deferred until the model/PPO milestone where SB3 becomes a declared dependency.

## Documentation

Update `f1rl/learning/README.md` to mark Gymnasium composition as implemented and PPO/model integration as the next layer.

Update repository documentation only where required to reflect installed runtime dependencies and the completed architecture boundary. Do not claim PPO, vectorization, track randomization, or rendering support.

## Verification

Before the feature is considered complete, run fresh:

```text
full pytest suite
focused Gym environment tests
Python 3.12 compileall
Gymnasium env checker
source/dependency boundary scans
git diff --check
```

A clean result must also prove no sandbox/runtime production files were modified merely to accommodate Gymnasium.

## Post-feature integration sequence

After this feature is fully verified:

```text
feat/gymnasium-environment
    -> merge locally into feat/sanbox-physics-core
    -> verify core
    -> integrate verified core into main
    -> verify main
    -> create models/ppo-baseline from verified main
```

The PPO branch is a separate architectural milestone. It will introduce Stable-Baselines3/PyTorch, model construction, vectorized training, save/load, and smoke-training acceptance without moving Gym semantics into the algorithm layer.
