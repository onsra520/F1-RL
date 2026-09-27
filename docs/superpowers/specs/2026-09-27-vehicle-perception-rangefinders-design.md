# Deterministic Vehicle Perception & Rangefinder System Design

## Goal

Build a deterministic sandbox perception layer that lets a vehicle measure nearby solid obstacles through configurable local-frame rangefinders.

The layer returns raw physical measurements only. It does not normalize observations, calculate RL rewards, invoke policies, render rays, or own runtime stepping.

## Architectural choice

V1 adds analytic ray-vs-oriented-box geometry to the existing track volume and builds vehicle-mounted sensors above that primitive.

```text
VehicleState + TrackWorld
        |
        v
   SensorSuite
        |
        v
RangefinderReading tuple
        |
        v
future learning.observation
```

This keeps `sandbox.perception` reusable by human debugging, replay, evaluation, and RL without making the sandbox know that a neural network exists.
## Scope

### In scope

- Deterministic analytic ray intersection with `OrientedBoxVolume`.
- Configurable vehicle-mounted rangefinder definitions.
- Raw rangefinder readings with distance and optional hit obstacle ID.
- A sensor suite that evaluates multiple rangefinders in declared order.
- Vehicle orientation rotating local sensor directions into world space.
- Nearest-obstacle selection with deterministic tie-breaking.

### Out of scope

- Observation normalization or tensor/NumPy conversion.
- PPO, Gymnasium, policy inference, reward shaping, or training.
- Road-edge or surface-boundary sensing.
- Boost, slow, checkpoint, finish, or pickup detection through rangefinders.
- Sensor noise, dropout, latency, stochasticity, or domain randomization.
- Sensor mount offsets or multiple ray origins; V1 rays originate at `vehicle.position`.
- Visualization/debug lines in Three.js.
- Swept vehicle collision or physics response.

## Package layout

```text
f1rl/sandbox/track/volume.py
f1rl/sandbox/perception/rangefinder.py
f1rl/sandbox/perception/sensor_suite.py
f1rl/sandbox/perception/__init__.py
```
## Ray/OBB geometry contract

`OrientedBoxVolume` gains:

```python
ray_distance(
    origin: Vec3,
    direction: Vec3,
    max_distance: float,
) -> float | None
```

Rules:

- `direction` must be unit length within absolute tolerance `1e-9`.
- `max_distance` must be finite and non-negative.
- The ray is half-infinite from `origin` along positive `direction`, then clipped to `max_distance`.
- If `origin` is inside or exactly on the inclusive box boundary, return `0.0`.
- A hit exactly at `max_distance` is included.
- A hit beyond `max_distance` returns `None`.
- Tangential/boundary contact counts as a hit.
- No sampling or stepping along the ray is allowed.

Implementation uses analytic slab intersection after expressing the world-space ray in the OBB's local right/up/forward basis. This reuses the same orientation convention as `contains()` and avoids needing a quaternion inverse API.

For a ray parallel to one local slab axis, that axis rejects the ray only when the origin component lies outside the slab. Otherwise that axis imposes no entry/exit bound.## Rangefinder definition

`f1rl/sandbox/perception/rangefinder.py` owns:

```python
@dataclass(frozen=True, slots=True)
class RangefinderSpec:
    id: str
    local_direction: Vec3
    max_distance: float
```

Validation:

- `id` must be a non-empty string after stripping whitespace.
- `local_direction` must be unit length within absolute tolerance `1e-9`.
- `max_distance` must be finite and strictly positive.

Local axes follow the existing vehicle convention:

```text
+x = right
+y = up
+z = forward
```

V1 does not normalize arbitrary input directions implicitly. Invalid non-unit directions fail fast so sensor geometry is explicit and reproducible.## Rangefinder reading

The same module owns the immutable raw measurement:

```python
@dataclass(frozen=True, slots=True)
class RangefinderReading:
    sensor_id: str
    distance: float
    hit_id: str | None
```

Semantics:

- A hit returns the exact analytic distance from the ray origin to the first obstacle boundary.
- If no obstacle is hit within range, `distance == spec.max_distance` and `hit_id is None`.
- If the ray origin is already inside/on an obstacle, `distance == 0.0` and `hit_id` identifies that obstacle.
- `distance` is always finite and within `[0, spec.max_distance]`.

The reading does not contain normalized values, rewards, collision flags, or rendering metadata.## Single-sensor sensing

`rangefinder.py` exposes:

```python
sense_rangefinder(
    spec: RangefinderSpec,
    track: TrackWorld,
    vehicle: VehicleState,
) -> RangefinderReading
```

Authoritative behavior:

1. Use `vehicle.position` as the ray origin.
2. Compute `world_direction = vehicle.orientation.rotate(spec.local_direction)`.
3. Test only `track.obstacles` using each obstacle volume's `ray_distance(...)`.
4. Ignore all non-solid track entities.
5. Choose the smallest hit distance.
6. If two or more obstacles hit at equal distance, choose lexical `obstacle.id`.
7. Return max-range/no-hit when there is no candidate.

No `TrackState`, runtime state, score, interaction state, or learning state is needed to sense obstacles.## Sensor suite

`f1rl/sandbox/perception/sensor_suite.py` owns:

```python
@dataclass(frozen=True, slots=True)
class SensorSuite:
    rangefinders: tuple[RangefinderSpec, ...]

    def sense(
        self,
        track: TrackWorld,
        vehicle: VehicleState,
    ) -> tuple[RangefinderReading, ...]: ...
```

Construction rules:

- `rangefinders` must be a tuple.
- Sensor IDs must be globally unique within the suite.
- The tuple's declared order is preserved exactly.

`sense()` returns one reading per spec in that same declared order. Sensor order is configuration data because future observation encoding will map this stable order into a vector.## Determinism and immutability

For identical `TrackWorld`, `VehicleState`, and `SensorSuite` inputs, repeated sensing must return exactly equal readings in exactly equal order.

Perception code must not:

- mutate track entities, vehicle state, or sensor definitions;
- read wall-clock time, random values, environment state, networking state, or rendering state;
- depend on collection iteration order for tie-breaking.

Track obstacles are already canonicalized by ID, but nearest-hit selection still explicitly compares `(distance, obstacle.id)` so the contract remains deterministic even if a future caller supplies an equivalent obstacle sequence differently.

## Dependency boundary

```text
sandbox.track geometry
        ^
        |
sandbox.perception
        ^
        |
future learning.observation / runtime debugging
```

`sandbox.perception` may depend on sandbox geometry, vehicle state, obstacle entities, and TrackWorld. It must not import `f1rl.learning`, `f1rl.runtime`, `f1rl.api`, drivers, Gymnasium, PPO, FastAPI, WebSocket, React, or Three.js.

Normalization belongs in `f1rl.learning.observation`; visualization belongs above the sandbox.## Verification criteria

The subsystem is ready for observation-layer integration when tests prove:

1. Axis-aligned and rotated `OrientedBoxVolume` rays return exact analytic hit distances.
2. Rays starting inside/on a volume return `0.0`.
3. Parallel, tangential, max-range, miss, and beyond-range cases obey the geometry contract.
4. Invalid non-unit directions and invalid ranges fail deterministically.
5. A forward sensor rotates with vehicle orientation rather than staying in world coordinates.
6. A no-hit sensor reports its configured max distance with `hit_id is None`.
7. Multiple obstacles choose the nearest hit; equal-distance hits choose lexical obstacle ID.
8. Boost, slow, checkpoint, finish, and score-pickup volumes do not appear as rangefinder hits.
9. `SensorSuite` rejects duplicate IDs and preserves declared sensor order exactly.
10. Repeated identical sensing returns exactly equal reading tuples.
11. Sensing does not mutate vehicle, track, obstacle, or sensor objects.
12. Existing physics, TrackWorld, world-step, and runtime suites remain green.
13. Sandbox perception has no learning/runtime/API/rendering dependencies.

## Next milestone boundary

After raw sensing is stable, `f1rl.learning.observation` can convert sandbox values into a fixed neural-network observation layout, for example:

```text
ordered rangefinder distances
+ vehicle velocity
+ heading/orientation features
+ checkpoint direction/distance
+ grounded state
        |
        v
normalized observation vector
        |
        v
Gymnasium / PPO
```

The perception layer itself remains raw and model-agnostic.