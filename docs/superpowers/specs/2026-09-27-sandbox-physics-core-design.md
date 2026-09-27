# Sandbox Physics Core Design

## Goal

Build the first deterministic vertical slice of `f1rl.sandbox` as a simplified 3D vehicle-motion kernel. The core must support sloped track surfaces, ramps, loss of ground contact, airtime under gravity, and landing while remaining small enough for RL training and unit testing.

The physics core is independent of Gymnasium, PPO, FastAPI, Three.js, rendering, scoring, and gameplay presentation.

## Scope

### In scope

- Immutable 3D geometry values for vectors and orientation.
- Immutable vehicle state, controls, and physical parameters.
- Separate grounded and airborne motion models.
- Gravity-driven airborne motion.
- Surface-aware grounded motion on slopes and ramps.
- Ground/air transitions driven by a surface-contact sample.
- Deterministic numerical integration over a caller-supplied `dt`.
- Forward-focused arcade vehicle behavior suitable for RL rollouts.

### Out of scope

- Per-wheel rigid-body simulation, suspension springs, tire slip curves, tire temperature, gearbox, differential, downforce, traction control, or F1-grade dynamics.
- Collision geometry beyond the ground-contact abstraction required by this milestone.
- Obstacles, boost pads, sensors, rewards, Gymnasium adapters, policy code, rendering, or runtime wall-clock time.
## Coordinate and unit conventions

- World coordinates use `(x, y, z)`.
- `+y` is world up, matching the intended Three.js presentation convention.
- Distances are meters, time is seconds, velocity is meters/second, acceleration is meters/second squared.
- Vehicle local axes are `+x` right, `+y` up, and `+z` forward.
- Vehicle orientation is represented internally by a normalized quaternion that rotates those local axes into world space.
- Debug/UI layers may derive yaw, pitch, and roll, but Euler angles are not the source of truth.
- Gravity defaults to `Vec3(0, -9.81, 0)`.
- `steering` is normalized to `[-1, 1]`.
- `throttle` and `brake` are normalized to `[0, 1]`.

## Geometry model

`f1rl/sandbox/geometry/vector.py` owns an immutable `Vec3` value with the minimum operations required by the physics core: addition, subtraction, scalar multiplication/division, dot product, cross product, magnitude, normalization, and projection/rejection against a normal.

`f1rl/sandbox/geometry/rotation.py` owns an immutable normalized `Quaternion` value with identity construction, multiplication, vector rotation, normalization, and construction from an axis-angle or orthonormal basis.

These are project-domain values, not renderer types. The sandbox must not import Three.js, NumPy, or a game-engine math library.

## Vehicle domain model

`f1rl/sandbox/vehicle.py` owns frozen dataclasses for `VehicleState`, `VehicleControl`, and `VehicleParameters`.
```python
VehicleState(
    position: Vec3,
    velocity: Vec3,
    orientation: Quaternion,
    angular_velocity: Vec3,
    grounded: bool,
)

VehicleControl(
    steering: float,
    throttle: float,
    brake: float,
)

VehicleParameters(
    wheelbase: float,
    ride_height: float,
    max_steering_angle: float,
    max_acceleration: float,
    max_braking: float,
    rolling_drag: float,
    air_drag: float,
    max_ground_speed: float,
    gravity: float,
)
```

`VehicleState.speed` is a derived property equal to `velocity.magnitude`; scalar speed is not stored independently.

Validation rules: normalized controls must be in range; wheelbase, ride height, and gravity must be positive; physical limits and drag coefficients must be non-negative; `max_steering_angle < pi / 2`; orientation must be normalizable. Invalid values raise `ValueError` rather than being silently clipped.

## Surface-contact contract

`f1rl/sandbox/surface.py` owns the presentation-neutral contact value passed into dynamics:

```python
SurfaceContact(
    point: Vec3,
    normal: Vec3,
)
```

`point` and `normal` define the local supporting surface plane near the vehicle. The normal must be unit length and point away from the drivable surface. `None` means the vehicle has no supporting surface and is airborne.

Track/world code will eventually produce this value from real geometry. Physics tests may use synthetic contacts such as a flat plane or an inclined ramp, keeping dynamics independent of track representation.

## Ground dynamics

`f1rl/sandbox/dynamics/ground_dynamics.py` computes grounded motion from vehicle state, controls, parameters, and `SurfaceContact`.

- Vehicle forward is projected onto the surface tangent plane and normalized.
- Longitudinal acceleration combines throttle, braking, rolling drag, and the component of gravity tangent to the surface.
- Ground velocity is constrained to the surface tangent plane.
- Ground speed is capped by `max_ground_speed`; airborne speed is not clamped by this rule.
- Steering uses the kinematic bicycle yaw-rate approximation around the current surface normal.
- Ground orientation aligns the vehicle up axis with the surface normal while preserving the steering-defined forward direction.

This means an inclined ramp naturally produces upward velocity. Losing contact preserves that 3D velocity; no artificial jump impulse is required.
## Air dynamics

`f1rl/sandbox/dynamics/air_dynamics.py` computes airborne motion when no `SurfaceContact` exists.

- Linear acceleration is gravity plus simple linear air drag.
- Steering, throttle, and brake do not directly change airborne trajectory in V1.
- Linear velocity remains fully 3D.
- Angular velocity persists through airtime.
- Orientation advances from angular velocity rather than being forced to the horizon or track.

The model intentionally supports ballistic jumps without implementing aerodynamic downforce or active air control.

## Vehicle dynamics coordinator

`f1rl/sandbox/dynamics/vehicle_dynamics.py` exposes the single high-level physics entry point for derivative calculation. It selects grounded or airborne dynamics from the supplied contact and returns a common `MotionDerivatives` value containing linear acceleration and angular velocity information required by integration.

The caller does not call ground/air modules directly during normal simulation.

The common result is:

```python
MotionDerivatives(
    linear_acceleration: Vec3,
    angular_velocity: Vec3,
)
```

`vehicle_dynamics.py` exposes:

```python
compute_motion(
    state: VehicleState,
    control: VehicleControl,
    parameters: VehicleParameters,
    contact: SurfaceContact | None,
) -> MotionDerivatives
```

## Integration and transitions

`f1rl/sandbox/dynamics/integrator.py` advances state with deterministic semi-implicit Euler integration:

```text
velocity_next = velocity + acceleration * dt
position_next = position + velocity_next * dt
```

Orientation advances by applying the quaternion rotation implied by angular velocity over `dt` and is re-normalized.

When a surface contact exists, the integrator applies the ground constraint after motion: the integrated chassis reference position is projected along the contact normal onto the local surface plane while preserving tangential displacement, then offset by `ride_height`; inward normal velocity is removed, and the resulting state is marked grounded. When contact is absent, the resulting state is marked airborne and no ground projection is applied.

A newly airborne state therefore retains ramp-generated vertical velocity. A newly landed state removes the velocity component directed into the surface; V1 does not bounce.
## Dependency boundary

Allowed dependency direction:

```text
geometry/
   ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
vehicle.py + surface.py
   ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
dynamics/ground_dynamics.py + dynamics/air_dynamics.py
   ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
dynamics/vehicle_dynamics.py
   ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
dynamics/integrator.py
```

All physics modules use Python standard-library math only. They must not import learning, runtime, API, web, Three.js, or a third-party physics engine.

## Verification criteria

The core is ready for world/collision/perception work when tests prove:

1. `Vec3` and quaternion operations are deterministic and preserve required normalization invariants.
2. Flat-ground throttle accelerates the vehicle along its forward direction.
3. Braking and rolling drag reduce grounded forward speed without creating reverse motion.
4. Steering changes heading around the surface normal with the expected sign.
5. A sloped surface gives grounded velocity a matching vertical component.
6. Removing contact after a ramp preserves the launch velocity and gravity decreases vertical velocity over time.
7. Airborne motion follows a deterministic ballistic trajectory with drag.
8. Landing removes inward normal velocity and marks the vehicle grounded without a bounce.
9. `max_ground_speed` applies only while grounded.
10. Identical state/control/contact/parameters/`dt` produce identical output.
11. Invalid controls, geometry values, parameters, and non-positive `dt` fail explicitly.
12. Physics state and domain inputs are never mutated in place.
