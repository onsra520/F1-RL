# Sandbox Physics Core Design

## Goal

Build the first deterministic vertical slice of `f1rl.sandbox`: a small 2D vehicle-motion kernel that converts normalized driver controls into the next vehicle state. This kernel is independent of Gymnasium, PPO, FastAPI, Three.js, rendering, collision rules, and track geometry.

## Scope

### In scope

- Immutable domain values for vehicle state, controls, and physical parameters.
- A kinematic bicycle approximation for longitudinal acceleration and yaw rate.
- Explicit-Euler integration over a caller-supplied `dt`.
- Forward-only motion for V1.
- Deterministic, pure state transitions suitable for unit testing and RL rollouts.

### Out of scope

- Reverse gear, gearbox, tire slip, tire temperature, suspension, downforce, aerodynamics, traction control, and F1-grade vehicle dynamics.
- Collision detection, track bounds, obstacles, boost pads, sensors, rewards, Gymnasium adapters, or policy code.
- Rendering and runtime wall-clock time.

## Coordinate and control conventions

- World plane: Cartesian `(x, y)`.
- Heading `0 rad` points along `+x`.
- Positive heading rotates counter-clockwise.
- Distances are meters, time is seconds, speed is meters/second.
- `steering` is normalized to `[-1, 1]`.
- `throttle` and `brake` are normalized to `[0, 1]`.
- V1 speed is clamped to `[0, max_speed]`; the vehicle cannot reverse.

## Domain model

`f1rl/sandbox/vehicle.py` owns three frozen dataclasses:

```python
VehicleState(x: float, y: float, heading: float, speed: float)
VehicleControl(steering: float, throttle: float, brake: float)
VehicleParameters(
    wheelbase: float,
    max_steering_angle: float,
    max_acceleration: float,
    max_braking: float,
    rolling_drag: float,
    max_speed: float,
)
```

Validation rules:

- `VehicleState.speed >= 0`.
- `VehicleControl.steering` must be in `[-1, 1]`.
- `VehicleControl.throttle` and `brake` must be in `[0, 1]`.
- All vehicle parameters must be non-negative and `wheelbase > 0`.
- `max_steering_angle < pi / 2` to avoid the tangent singularity.

Invalid domain values raise `ValueError`; inputs are not silently clipped.

## Dynamics

`f1rl/sandbox/dynamics/vehicle_dynamics.py` exposes a pure function:

```python
compute_derivatives(
    state: VehicleState,
    control: VehicleControl,
    parameters: VehicleParameters,
) -> MotionDerivatives
```

with:

```python
MotionDerivatives(dx: float, dy: float, dheading: float, dspeed: float)
```

Equations:

```text
steering_angle = control.steering * max_steering_angle
acceleration = throttle * max_acceleration
             - brake * max_braking
             - rolling_drag * speed

dx       = speed * cos(heading)
dy       = speed * sin(heading)
dheading = speed / wheelbase * tan(steering_angle)
dspeed   = acceleration
```

This is intentionally a simple kinematic model. It gives the RL layer continuous steering/throttle behavior without introducing a high-fidelity racing simulator into the first milestone.

## Integration

`f1rl/sandbox/dynamics/integrator.py` exposes:

```python
integrate(
    state: VehicleState,
    derivatives: MotionDerivatives,
    parameters: VehicleParameters,
    dt: float,
) -> VehicleState
```

Rules:

- `dt` must be strictly positive.
- Explicit Euler advances every component by `derivative * dt`.
- Resulting speed is clamped to `[0, parameters.max_speed]`.
- Heading is normalized to `[-pi, pi)`.
- The input objects are never mutated.

## Dependency boundary

Allowed dependency direction:

```text
vehicle.py <- dynamics/vehicle_dynamics.py <- dynamics/integrator.py
```

Both dynamics modules may use Python standard-library math only. They must not import learning, runtime, API, or web concerns.

## Verification criteria

The core is ready for the next milestone when tests prove:

1. Zero steering at constant speed produces straight-line motion.
2. Throttle increases speed.
3. Brake decreases speed but integration never produces reverse motion.
4. Steering changes heading with the expected sign.
5. Maximum speed is enforced by integration.
6. Identical state/control/parameters/`dt` produce identical output.
7. Invalid control, parameter, state, and `dt` values fail explicitly.
