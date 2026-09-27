# Deterministic Local-Frame Observation Encoding Design

## Goal

Build a deterministic learning-layer encoder that converts raw sandbox perception, vehicle state, and an explicitly supplied navigation target into a fixed normalized numeric observation tuple suitable for a future Gymnasium/PPO adapter.

The encoder must remain model-agnostic. It does not choose route progress, calculate rewards, map actions, step the simulation, or depend on NumPy/Gymnasium/PyTorch.

## Architectural choice

V1 keeps target-selection policy outside the encoder:

```text
TrackWorld + VehicleState ----> SensorSuite ----> raw rangefinder readings
             |                                      |
             |                                      v
             +-------------> ObservationEncoder <---+
                                  ^
                                  |
                         NavigationTarget | None
                                  |
                                  v
                         tuple[float, ...]
```
This boundary is intentional: the current sandbox tracks checkpoint overlap events but does not yet own completed-checkpoint/lap progress. `learning.observation` must therefore never infer a "next checkpoint" from `TrackWorld.checkpoints`.

## Scope

### In scope

- Immutable observation configuration with explicit normalization scales.
- Immutable `NavigationTarget(position: Vec3)` input DTO.
- Ordered normalized rangefinder features from the configured `SensorSuite`.
- Vehicle linear velocity projected into the vehicle-local right/up/forward basis.
- Navigation-target direction projected into the same local basis.
- Normalized target distance plus explicit target-availability flag.
- Grounded flag.
- Stable feature names and deterministic observation size.
- Deterministic pure-Python tuple output.

### Out of scope

- Selecting the next checkpoint or tracking race/lap progress.
- Absolute world position or absolute yaw/quaternion features.
- Angular velocity.
- Reward shaping, scoring, action mapping, Gymnasium, PPO, training, or evaluation orchestration.
- NumPy, tensors, observation spaces, batching, frame stacking, recurrence, or history.
- Sensor noise/domain randomization.
## Package layout

```text
f1rl/learning/__init__.py
f1rl/learning/observation.py
f1rl/learning/README.md
tests/learning/test_observation.py
tests/integration/test_observation_encoding.py
```

No sandbox file changes are required for this milestone.

## Configuration contract

`f1rl/learning/observation.py` owns:

```python
@dataclass(frozen=True, slots=True)
class ObservationConfig:
    sensor_suite: SensorSuite
    velocity_scale: float
    target_distance_scale: float
```

Validation:

- `sensor_suite` must be a `SensorSuite`.
- `velocity_scale` must be numeric, finite, strictly positive, and must reject booleans.
- `target_distance_scale` follows the same rules.
- Configuration is immutable after construction.
## Navigation target contract

The same module owns:

```python
@dataclass(frozen=True, slots=True)
class NavigationTarget:
    position: Vec3
```

`NavigationTarget` is an encoder input DTO, not race-progress state. A future progress/session/Gym layer decides which target position is currently authoritative and constructs this DTO at the observation boundary.

The encoder accepts `NavigationTarget | None`. `None` means there is currently no navigation target.

## Encoder contract

```python
@dataclass(frozen=True, slots=True)
class ObservationEncoder:
    config: ObservationConfig

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def size(self) -> int: ...

    def encode(
        self,
        track: TrackWorld,
        vehicle: VehicleState,
        target: NavigationTarget | None,
    ) -> tuple[float, ...]: ...
```
`ObservationEncoder` is immutable and stateless. Repeated calls with equal inputs return equal tuples.

## Feature layout

For `N` configured rangefinders, observation size is exactly `N + 9`.

Features appear in this order:

```text
0 .. N-1      normalized rangefinder distances, SensorSuite declaration order
N             velocity_local_x
N + 1         velocity_local_y
N + 2         velocity_local_z
N + 3         target_direction_local_x
N + 4         target_direction_local_y
N + 5         target_direction_local_z
N + 6         target_distance
N + 7         target_available
N + 8         grounded
```

The rangefinder prefix must preserve the exact `SensorSuite.rangefinders` order. It must never sort by sensor ID.

For eight sensors this produces 17 features.

All emitted values are finite and bounded to `[-1.0, 1.0]`. Rangefinder distances, normalized target distance, and boolean flags use the narrower `[0.0, 1.0]` range.
## Rangefinder encoding

`encode(...)` obtains raw readings through the configured `SensorSuite`:

```python
readings = config.sensor_suite.sense(track, vehicle)
```

Each reading is normalized against the corresponding spec in the same tuple position:

```python
normalized = reading.distance / spec.max_distance
```

Contract:

- a zero-distance obstacle emits `0.0`;
- a miss emits exactly `1.0` because perception returns `distance == max_distance`;
- intermediate physical distances map linearly into `(0.0, 1.0)`;
- hit IDs are intentionally not encoded;
- rangefinder feature order follows declaration order, not hit order or lexical ID order.

The encoder may defensively clamp the quotient to `[0.0, 1.0]` for floating-point boundary noise, but it must not alter the underlying perception objects.

## Vehicle-local basis

Use the existing vehicle convention:

```text
+x = right
+y = up
+z = forward
```
World-space basis vectors are derived from vehicle orientation:

```python
right = vehicle.orientation.rotate(Vec3(1.0, 0.0, 0.0))
up = vehicle.orientation.rotate(Vec3(0.0, 1.0, 0.0))
forward = vehicle.orientation.rotate(Vec3(0.0, 0.0, 1.0))
```

Any world vector `v` is expressed in vehicle-local coordinates with dot products:

```python
local = Vec3(v.dot(right), v.dot(up), v.dot(forward))
```

No Euler-angle extraction or quaternion inversion is required.

## Velocity encoding

Vehicle linear velocity is projected into the local basis, divided component-wise by `config.velocity_scale`, then clamped independently into `[-1.0, 1.0]`.

```text
local velocity component = dot(world velocity, local axis)
normalized component     = clamp(component / velocity_scale, -1, +1)
```

This preserves reverse/lateral/vertical motion signs while preventing unusually large airborne/collision velocities from escaping the observation bounds.
## Navigation-target encoding

When a target exists:

```python
delta = target.position - vehicle.position
distance = delta.magnitude
```

If `distance > 0`, normalize `delta` to a world direction and project that unit vector into the vehicle-local right/up/forward basis. These three components already lie within `[-1.0, 1.0]` up to floating-point noise and may be defensively clamped to that interval.

Target distance is encoded as:

```python
min(distance / config.target_distance_scale, 1.0)
```

and `target_available = 1.0`.

If the vehicle is exactly at the target position, direction features are `(0.0, 0.0, 0.0)`, target distance is `0.0`, and `target_available` remains `1.0`.

If `target is None`, direction features are `(0.0, 0.0, 0.0)`, target distance is `0.0`, and `target_available = 0.0`.
## Grounded flag

The final feature is:

```python
1.0 if vehicle.grounded else 0.0
```

No other vehicle-state boolean is encoded in V1.

## Feature names

`feature_names` is deterministic and derived only from immutable configuration:

```text
rangefinder:<sensor-id>          # one per configured sensor, declaration order
velocity_local_x
velocity_local_y
velocity_local_z
target_direction_local_x
target_direction_local_y
target_direction_local_z
target_distance
target_available
grounded
```

`size == len(feature_names)` and must also equal `len(config.sensor_suite.rangefinders) + 9`.

Changing sensor count/order/IDs changes the observation contract and therefore defines a different model input layout. The encoder must never silently reorder an existing layout.
## Rotation-invariance objective

V1 intentionally omits absolute world position and absolute orientation features.

Rangefinder directions, velocity, and target direction are all vehicle-relative. Therefore two equivalent scenes related by the same rigid yaw rotation should produce equivalent observations within normal floating-point tolerance.

This property is deliberate for generalization to unseen user-created tracks: the model should learn relationships such as "target is ahead-right" rather than memorize a global compass heading.

The encoder does not guarantee invariance to translation/rotation when the physical scene itself changes relative to the vehicle, only when vehicle, obstacles, velocity, and target are transformed consistently.

## Determinism and immutability

For equal `ObservationConfig`, `TrackWorld`, `VehicleState`, and `NavigationTarget | None` inputs, repeated encoding must return exactly equal tuples.

Encoding must not mutate:

- `TrackWorld` or obstacle entities;
- `VehicleState` or its vectors/quaternion;
- `SensorSuite` or rangefinder specs;
- `NavigationTarget`;
- configuration or encoder state.

No wall clock, randomness, environment variables, network state, rendering state, or process-global mutable state may influence the result.
## Dependency boundary

```text
sandbox geometry / vehicle / perception / track
                    ^
                    |
          learning.observation
                    ^
                    |
          future Gym / PPO / evaluation
```

`f1rl.learning.observation` may depend on sandbox domain/perception types and Python standard library only.

It must not import or depend on:

- `f1rl.runtime` or session scheduling;
- `f1rl.api` or drivers;
- Gymnasium;
- NumPy;
- PyTorch;
- Stable-Baselines3/PPO;
- FastAPI/WebSocket;
- React/Three.js/rendering code.

Sandbox must not import `f1rl.learning`.
## Verification criteria

The milestone is complete when tests prove:

1. `ObservationConfig` rejects invalid sensor-suite/scales and is immutable.
2. `NavigationTarget` is immutable and preserves the supplied `Vec3`.
3. Observation size is exactly `sensor_count + 9` and equals `len(feature_names)`.
4. Rangefinder features preserve declaration order and normalize each reading by its own configured max distance.
5. No-hit rangefinder readings encode exactly `1.0`; zero-distance hits encode `0.0`.
6. Vehicle velocity is expressed in local right/up/forward axes, scaled, and clamped to `[-1, 1]`.
7. Target direction is vehicle-relative, including rotated-vehicle cases.
8. Target distance is normalized/clamped to `[0, 1]`.
9. `target is None` and `vehicle exactly at target` remain distinguishable through `target_available`.
10. Grounded encodes exactly `0.0` or `1.0`.
11. Equal inputs produce exactly equal tuples and no input object is mutated.
12. Equivalent consistently yaw-rotated scenes produce approximately equal observations.
13. Output values are finite and inside declared bounds.
14. Existing sandbox, runtime, and perception test suites remain green.
15. Learning observation has no runtime/Gym/NumPy/PPO/API/rendering dependencies, and sandbox does not import learning.

## Next milestone boundary

After this encoder is stable, the next layer can add deterministic action mapping from policy outputs into `VehicleControl`, followed by a Gymnasium adapter that composes:

```text
ObservationEncoder + action mapping + Simulation.step() + reward composer
```

Race-progress/next-checkpoint selection remains a separate concern and can later provide the authoritative `NavigationTarget` without changing the observation encoder's responsibility.
