# Environment Protocol Freeze Design

**Status:** Approved conversational design; written specification for review.

**Branch:** `models/ppo-baseline`

**Base HEAD at spec creation:** `7aafb5f9eb57a07898b05b68aeeb50897b6e94da`

## Goal

WP1B resolves the environment-level research values intentionally left open by WP1A after WP2 established a concrete, hashable six-track corpus.

The outcome is a single balanced environment protocol for `ppo-generalization-v1` covering:

- vehicle dynamics;
- simulation timestep and surface-query tolerance;
- rangefinder layout and range;
- observation normalization scales;
- reward weights;
- episode horizon.

These values become the pinned environment contract for protocol v1. They are selected from the implemented dynamics and the geometry scale of the authored corpus, not promoted from historical test fixtures.

WP1B does **not** freeze the entire experiment. Training budget/backend/profile and the final replicate seeds remain owned by WP7/WP8, so `experiment.status` remains `draft` and `training.protocol_status` remains `pilot_owned` after WP1B.

## Research framing

The experiment continues to study:

> **Generalization of a PPO waypoint-guided navigation/control policy to held-out track geometries.**

The policy receives local rangefinder distances plus local waypoint direction and distance. WP1B does not change that semantic contract and does not claim raw unseen-map discovery.

## Evidence base

WP2 provides six candidate tracks with road widths of 12–16 m, route scales of roughly 108–149 m, 90-degree planar turns, one 8-degree elevated ramp, and slalom obstacles offset from the centerline.

The relevant waypoint-polyline scales are:

| Track | Approximate route length | Longest target segment |
| --- | ---: | ---: |
| `straight-sprint-v1` | 108.00 m | 40.00 m |
| `slalom-v1` | 121.39 m | 40.45 m |
| `right-turn-v1` | 115.31 m | 44.00 m |
| `left-turn-v1` | 115.31 m | 44.00 m |
| `ramp-v1` | 149.30 m | 70.84 m |
| `offset-slalom-v1` | 133.12 m | 36.58 m |

These figures are derived from the authored spawn/checkpoint/finish centers and are protocol-design evidence, not evaluation metrics.

## Chosen approach

Use a balanced protocol rather than either a very conservative low-speed environment or an aggressive high-speed environment.

The balanced protocol aims to satisfy four constraints simultaneously:

1. vehicle turning geometry must fit the 12 m minimum road width;
2. one simulation step at maximum speed must remain smaller than the thinnest important interaction volumes;
3. sensor range must provide useful obstacle reaction time without making the observation unnecessarily global;
4. the episode horizon must allow substantial recovery/exploration without making stalled episodes excessively long.

## Canonical environment values

The checked-in experiment definition should resolve the environment section to the following values.

```yaml
environment:
  vehicle:
    wheelbase: 3.0
    ride_height: 0.5
    max_steering_angle: 0.5
    max_acceleration: 6.0
    max_braking: 10.0
    rolling_drag: 0.4
    air_drag: 0.1
    max_ground_speed: 15.0
    gravity: 9.81

  simulation:
    fixed_dt: 0.1
    max_surface_distance: 0.75

  observation:
    velocity_scale: 15.0
    target_distance_scale: 80.0
    rangefinders:
      - id: left-30deg
        local_direction: [-0.5, 0.0, 0.8660254037844386]
        max_distance: 30.0
      - id: left-15deg
        local_direction: [-0.25881904510252074, 0.0, 0.9659258262890683]
        max_distance: 30.0
      - id: left-7p5deg
        local_direction: [-0.13052619222005157, 0.0, 0.9914448613738104]
        max_distance: 30.0
      - id: front
        local_direction: [0.0, 0.0, 1.0]
        max_distance: 30.0
      - id: right-7p5deg
        local_direction: [0.13052619222005157, 0.0, 0.9914448613738104]
        max_distance: 30.0
      - id: right-15deg
        local_direction: [0.25881904510252074, 0.0, 0.9659258262890683]
        max_distance: 30.0
      - id: right-30deg
        local_direction: [0.5, 0.0, 0.8660254037844386]
        max_distance: 30.0

  reward:
    target_progress:
      weight_per_meter: 1.0
    collision:
      penalty: 5.0
    time:
      penalty_per_second: 0.1

  episode:
    max_steps: 500
```

The declaration order of rangefinders is part of the protocol because observation features preserve sensor declaration order.

## Vehicle protocol rationale

### Wheelbase and steering

Ground yaw rate is implemented as:

```text
yaw_rate = forward_speed / wheelbase * tan(steering_angle)
```

With:

```text
wheelbase = 3.0 m
max_steering_angle = 0.5 rad
```

the idealized minimum bicycle-model turn radius is:

```text
R = 3.0 / tan(0.5) ~= 5.49 m
```

The narrowest authored roads are 12 m wide, giving a nominal half-width of 6 m. The selected steering authority is therefore strong enough to negotiate the corpus's 90-degree turns without making the vehicle behave like an unrealistically short-wheelbase pivoting agent.

### Acceleration, drag, and speed

Use:

```text
max_acceleration = 6.0 m/s^2
rolling_drag = 0.4 1/s
max_ground_speed = 15.0 m/s
```

At full throttle on a neutral flat surface, the longitudinal model gives:

```text
6.0 - 0.4 * 15.0 = 0
```

so the drag equilibrium aligns with the explicit ground-speed cap. This gives one coherent nominal top-speed scale instead of two unrelated limiting mechanisms.

`max_ground_speed = 15.0 m/s` is intentionally moderate. With `fixed_dt = 0.1 s`, the maximum nominal translation is approximately 1.5 m per tick. This is below the 2 m total thickness of the thinnest checkpoint/finish volumes and below the 4 m obstacle thickness in the slalom tracks, reducing event/obstacle tunneling risk in the discrete simulation.

### Braking

`max_braking = 10.0 m/s^2` is frozen for completeness of `VehicleParameters`, but it is behaviorally dormant in protocol v1 because the action contract remains steering + throttle with `brake = 0`.

Changing the action contract to expose braking is a material protocol change; WP1B does not do that.

### Gravity and air drag

Use conventional gravity:

```text
gravity = 9.81 m/s^2
```

and modest linear air drag:

```text
air_drag = 0.1 1/s
```

The ramp remains a true 3D case while airborne motion continues to be dominated by gravity rather than aggressive artificial damping.

### Ride height

Use:

```text
ride_height = 0.5 m
```

The integrator constrains a grounded vehicle to this offset above the detected surface. WP2 track assets intentionally store spawn pose independently of vehicle parameters, so the authored spawn position is not rewritten by WP1B.

All six candidate assets currently share the same spawn-height convention. Any first-step ride-height alignment is therefore common across the corpus rather than a track-specific initial-state confound.

## Simulation protocol rationale

### Fixed timestep

Use:

```text
fixed_dt = 0.1 s
```

This corresponds to a 10 Hz deterministic physics step and gives the 1.5 m/tick maximum nominal ground translation described above.

WP1B does not introduce frame-time coupling or variable timesteps.

### Surface query distance

Use:

```text
max_surface_distance = 0.75 m
```

This exceeds the 0.5 m ride-height offset while staying small relative to road and ramp geometry. The surface query therefore tolerates the normal grounded offset without turning distant geometry into a valid ground contact.

`max_surface_distance` remains an environment parameter rather than a track field.

## Sensor protocol

Use seven horizontal rangefinders ordered from left to right:

```text
-30 deg
-15 deg
-7.5 deg
  0 deg
+7.5 deg
+15 deg
+30 deg
```

Every sensor uses:

```text
max_distance = 30.0 m
```

The exact local unit vectors in the canonical YAML are part of the protocol and must not be recomputed with platform-dependent rounding at runtime.

### Why seven rays

Current rangefinders detect obstacles only; they do not detect road edges. Navigation direction is already supplied separately by waypoint target features.

The slalom corpora place obstacles approximately 3.5–4.5 m off the centerline. Narrow rays at 7.5 and 15 degrees provide useful forward-offset coverage, while 30-degree rays provide wider near-field coverage. Seven rays avoid expanding the observation dimensionality with many sensors that would mostly duplicate waypoint guidance or observe no additional entity class.

At the 15 m/s protocol speed, 30 m corresponds to roughly two seconds of nominal forward travel, giving useful reaction distance without exposing the entire 100+ m route.

## Observation normalization

### Velocity

Use:

```text
velocity_scale = 15.0
```

This equals `max_ground_speed`, so legal ground-speed magnitudes map naturally into the intended normalized range rather than saturating below the configured speed cap.

The encoder still clamps each local velocity component to `[-1, 1]`.

### Target distance

Use:

```text
target_distance_scale = 80.0 m
```

The longest current spawn/waypoint segment is approximately 70.84 m on `ramp-v1`. An 80 m scale therefore covers the existing corpus with margin while preserving more distance resolution than a 100–150 m normalization scale.

This is not a promise that every future user-authored track fits protocol v1. Tracks used by protocol v1 must satisfy the protocol's authored-corpus compatibility constraints; materially larger geometry belongs to a new protocol or requires an explicit compatibility review.

## Reward protocol

Use:

```text
target_progress.weight_per_meter = 1.0
collision.penalty = 5.0
time.penalty_per_second = 0.1
```

The unit interpretation is deliberately simple:

```text
+1 reward  ~= 1 m of progress toward the current target
-5 reward  = one obstacle collision
-1 reward  = 10 s of elapsed time
```

The intended ordering is:

```text
make route progress
  > avoid repeated collisions
  > prefer faster completion when progress/collision behavior is similar
```

At the 50 s episode horizon, the maximum accumulated time penalty is approximately `-5`. Time therefore discourages stalling without dominating the positive progress signal over a 100–150 m route.

Collision penalty is tied to actual `ObstacleHit` events. The obstacle resolver rejects penetration and stops the vehicle, so repeatedly driving into an obstacle naturally incurs repeated penalties rather than being treated as one persistent geometric state.

The numerical tuple `(1.0, 5.0, 0.1)` may resemble historical test fixtures, but WP1B's justification is the implemented reward units and the authored corpus scale. Test fixtures are not the source of these values.

## Episode horizon

Use:

```text
max_steps = 500
```

With `fixed_dt = 0.1 s`, this yields:

```text
500 * 0.1 = 50 s
```

The longest current waypoint polyline is approximately 149.30 m. Completing that route within 50 s requires only about 2.99 m/s average waypoint-path progress, substantially below the 15 m/s configured ground-speed ceiling.

The horizon therefore allows recovery and exploration during learning while still bounding stalled episodes.

WP1B does not add off-track termination, collision-count termination, or any new episode rule.

## Experiment lifecycle after WP1B

WP1B freezes the **environment portion** of protocol v1 but intentionally leaves the overall experiment artifact in development state:

```yaml
experiment:
  status: draft

training:
  protocol_status: pilot_owned
```

The seed arrays also remain unresolved until the later training protocol is selected.

Consequently, `ExperimentDefinition.require_frozen()` must continue to fail after WP1B. That is expected and protects the distinction between:

- environment protocol resolved by WP1B; and
- complete runnable scientific protocol frozen after WP7/WP8 decisions.

WP1B does not add a new `environment_status` field. Exact-value regression tests and the written protocol provide the environment freeze without changing the WP1A schema.

## Protocol-change rule

After WP1B is accepted, any change to the following is a material environment-protocol change:

- vehicle numeric parameters;
- fixed timestep;
- surface query distance;
- sensor count, IDs, order, direction, or range;
- observation scales;
- reward weights;
- episode maximum steps.

Such a change must not silently mutate the baseline after held-out results exist. It requires an explicit protocol review and, once the scientific baseline is frozen/used, a new protocol version.

## WP1B implementation scope

WP1B implementation should be deliberately small.

### In scope

- replace unresolved environment values in `configs/experiments/ppo-generalization-v1.yml` with the exact values in this specification;
- preserve `experiment.status: draft`;
- preserve `training.protocol_status: pilot_owned`;
- preserve unresolved final seed arrays;
- add focused regression tests that pin every environment value and sensor declaration order;
- prove the resolved values can instantiate the existing domain/runtime configuration types used by the environment;
- update experiment documentation with the environment-freeze rule and rationale;
- retain all existing dependency boundaries and tests.

### Out of scope

- selecting final PPO/sampling seeds;
- changing PPO hyperparameters;
- selecting final training timestep budget;
- selecting final vector backend or `n_envs`;
- changing `config.yml` training defaults;
- declaring the train/held-out split;
- changing or rehashing WP2 track assets;
- adding evaluation telemetry;
- implementing evaluation metrics;
- provenance collection;
- running real pilot training;
- changing action, observation, reward, or race semantics.

## Acceptance requirements

WP1B implementation must prove all of the following:

1. `ppo-generalization-v1.yml` loads through the existing strict `ExperimentDefinition` loader.
2. Every vehicle numeric value exactly matches this specification.
3. Simulation values are exactly `fixed_dt=0.1` and `max_surface_distance=0.75`.
4. The observation protocol contains exactly seven rangefinders in the specified order with the exact IDs, vectors, and 30 m ranges.
5. `velocity_scale` is exactly 15.0 and `target_distance_scale` exactly 80.0.
6. Reward values are exactly 1.0 progress-per-meter, 5.0 collision penalty, and 0.1 time penalty per second.
7. `max_steps` is exactly 500.
8. The resolved vehicle values instantiate `VehicleParameters` without changing its implementation.
9. The resolved simulation values instantiate `SimulationConfig` without changing its implementation.
10. The resolved sensor and observation values instantiate `RangefinderSpec`, `SensorSuite`, and `ObservationConfig` without changing their implementations.
11. The resolved reward values instantiate the existing three reward terms/composer without changing reward semantics.
12. The resolved episode value instantiates `EpisodeConfig` without changing episode semantics.
13. `experiment.status` remains `draft`.
14. `training.protocol_status` remains `pilot_owned`.
15. final seed arrays remain unresolved rather than inventing replicate seeds in WP1B.
16. `require_frozen()` still rejects the WP1B artifact for lifecycle reasons unrelated to unresolved environment numerics.
17. the WP2 manifest and all six track content/world hashes remain unchanged.
18. no SB3 or direct PyTorch production-import boundary expands.
19. no production mock/fake/test-only path is introduced.
20. the full existing test suite remains green.

## Next stage

After WP1B is implemented and verified, the research roadmap continues to WP3:

```text
WP1B Environment Protocol Freeze
  -> WP3 Train/Held-out Split + Leakage Guards
  -> WP4 Evaluation Telemetry
  -> WP5 Evaluator + Metrics
  -> WP6 Provenance
  -> WP7 Pilot Training
  -> freeze final training profile
  -> WP8 Baseline Generalization Experiment
```

WP3 may consume the now-pinned non-track environment contract but must not change it as part of split construction.
