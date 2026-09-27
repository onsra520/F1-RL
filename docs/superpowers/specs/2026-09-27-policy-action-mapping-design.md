# Strict 2D Policy Action Mapping Design

## Goal

Build a deterministic learning-layer action contract that converts a validated two-dimensional policy action into the existing sandbox `VehicleControl` used by `Simulation.step(...)`.

V1 intentionally exposes only steering and throttle. Braking remains outside the policy action space and is always mapped to `0.0`.

## Architectural choice

```text
future PPO / Gym / PolicyDriver
           |
           v
 PolicyAction(steering, throttle)
           |
           v
     learning.action
           |
           v
 VehicleControl(steering, throttle, brake=0)
           |
           v
    Simulation.step(control)
```

`learning.action` owns policy-facing action semantics. Sandbox continues to own physical control semantics.
## Scope

### In scope

- Immutable `PolicyAction(steering, throttle)` DTO.
- Strict validation of action values.
- Deterministic conversion to sandbox `VehicleControl`.
- Fixed action metadata for future adapters.
- Pure-Python output with no framework dependency.
- Tests for exact boundary values, invalid values, determinism, and dependency direction.

### Out of scope

- Policy-controlled braking.
- Signed longitudinal action semantics.
- Throttle remapping from `[-1, 1]` into `[0, 1]`.
- Action clamping or sanitizing invalid policy output.
- Reverse gear, handbrake, gears, traction control, or steering-rate limiting.
- Reward calculation, observation encoding, simulation stepping, Gymnasium, PPO, NumPy, or driver ownership.
## Package layout

```text
f1rl/learning/action.py
f1rl/learning/__init__.py
f1rl/learning/README.md
tests/learning/test_action.py
tests/integration/test_action_mapping.py
```

No sandbox or runtime production file changes are required.

## Policy action contract

`f1rl.learning.action` owns:

```python
@dataclass(frozen=True, slots=True)
class PolicyAction:
    steering: float
    throttle: float
```

Both values are normalized policy-space controls, not physical acceleration or wheel angles.
Validation rules:

- `steering` must be numeric, finite, and within `[-1.0, 1.0]` inclusive.
- `throttle` must be numeric, finite, and within `[0.0, 1.0]` inclusive.
- booleans are rejected even though Python treats `bool` as an `int` subtype.
- values outside the declared interval raise `ValueError`.
- no invalid value is clamped, wrapped, normalized, or otherwise rewritten.
- construction is immutable after validation.

Integers such as `0` or `1` are valid numeric inputs when they satisfy the range. The contract is semantic rather than dependent on the caller supplying a literal `float` object.

## Mapping contract

The module exposes:

```python
def map_policy_action(action: PolicyAction) -> VehicleControl:
    ...
```

The mapping is exact:

```text
VehicleControl.steering = PolicyAction.steering
VehicleControl.throttle = PolicyAction.throttle
VehicleControl.brake    = 0.0
```
The mapper requires an actual `PolicyAction`; passing another object type is a contract error and must raise `ValueError` rather than relying on incidental attribute access.

The mapper performs no additional scaling. In particular:

```text
policy throttle 0.0 -> control throttle 0.0
policy throttle 0.5 -> control throttle 0.5
policy throttle 1.0 -> control throttle 1.0
```

A neutral policy output must never become half throttle through a hidden `[-1,1] -> [0,1]` transformation.

## Action metadata

`f1rl.learning.action` exposes immutable tuple constants:

```python
ACTION_NAMES = ("steering", "throttle")
ACTION_LOW = (-1.0, 0.0)
ACTION_HIGH = (1.0, 1.0)
ACTION_SIZE = 2
```

These values are the authoritative V1 policy-action layout for future Gymnasium and policy adapters.
The metadata order is stable and must match the `PolicyAction` field order exactly. Future adapters may derive a framework-specific action space from these constants, but this module remains framework-agnostic.

## Brake semantics

V1 brake policy is intentionally fixed:

```python
brake = 0.0
```

This is not an omission inside the mapper; it is part of the public action contract. The current physics already supports braking through `VehicleControl`, but the learning policy does not control that degree of freedom yet.

Reasons:

- two policy dimensions reduce exploration/search complexity;
- throttle and brake cannot conflict because brake is fixed off;
- the project can first establish steering/throttle learning behavior;
- a future braking milestone can deliberately version/extend the action contract instead of silently changing V1 semantics.

Adding policy braking later is a model-input/output contract change and must not mutate V1 behavior in place.
## Determinism and immutability

For equal `PolicyAction` inputs, repeated mapping must return equal `VehicleControl` values.

Mapping must not mutate:

- the `PolicyAction` instance;
- sandbox `VehicleControl` definitions;
- any observation configuration or encoder state;
- runtime or simulation state.

The mapper is a pure conversion function. It uses no randomness, wall clock, environment variables, network state, process-global mutable state, or hidden history.

## Error handling

Validation is fail-fast at the policy boundary. Invalid policy values are programming/integration errors, not noisy physical inputs to sanitize.

Therefore V1 deliberately rejects rather than clamps:

```text
steering = 1.01  -> ValueError
throttle = -0.01 -> ValueError
throttle = 1.01  -> ValueError
nan / inf         -> ValueError
True / False      -> ValueError
```

## Integration acceptance

The acceptance test should prove the learning action contract composes with the existing deterministic runtime without introducing a production dependency on runtime.

Construct equivalent simulations and compare one step using:

```python
map_policy_action(PolicyAction(steering, throttle))
```

against one step using the exact direct sandbox control:

```python
VehicleControl(steering=steering, throttle=throttle, brake=0.0)
```

Equivalent initial states/configuration must produce equal simulation snapshots. This proves the mapper adds no hidden transformation.

Integration tests may import runtime; `f1rl.learning.action` itself may not.

## Public exports

`f1rl.learning.__init__` should export:

```text
PolicyAction
map_policy_action
ACTION_NAMES
ACTION_LOW
ACTION_HIGH
ACTION_SIZE
```

## Dependency boundary

```text
sandbox.VehicleControl
        ^
        |
 learning.action
        ^
        |
future Gym / PPO / PolicyDriver
```

`f1rl.learning.action` may depend on sandbox `VehicleControl` and Python standard library only.

It must not import or depend on:

- `f1rl.runtime` or simulation/session ownership;
- `f1rl.api` or `f1rl.drivers`;
- observation encoding or reward calculation;
- Gymnasium;
- NumPy;
- PyTorch;
- Stable-Baselines3/PPO;
- FastAPI/WebSocket;
- React/Three.js/rendering code.

Sandbox must not import `f1rl.learning`.
## Verification criteria

The milestone is complete when tests prove:

1. `PolicyAction` accepts exact lower/upper boundary values and ordinary in-range numeric values.
2. `PolicyAction` rejects booleans, non-numeric values, `nan`, infinities, and out-of-range steering/throttle.
3. `PolicyAction` is immutable after construction.
4. `ACTION_NAMES == ("steering", "throttle")` and ordering is stable.
5. `ACTION_LOW == (-1.0, 0.0)`, `ACTION_HIGH == (1.0, 1.0)`, and `ACTION_SIZE == 2`.
6. `map_policy_action(...)` rejects non-`PolicyAction` inputs.
7. Mapping copies steering and throttle exactly with no remapping or clamping.
8. Every mapped `VehicleControl` has `brake == 0.0` exactly.
9. Equal policy actions produce equal controls and mapping does not mutate the input.
10. A mapped control and the equivalent direct `VehicleControl` produce equal one-step simulation snapshots from equal initial conditions.
11. Existing sandbox/runtime/perception/observation tests remain green.
12. `learning.action` has no runtime/Gym/NumPy/PPO/API/driver/rendering dependency and sandbox does not import learning.

## Next milestone boundary

After this action mapper is stable, the next major layer can be a Gymnasium environment adapter that composes existing primitives rather than redefining them:
```text
ObservationEncoder
      +
PolicyAction mapping
      +
Simulation.step()
      +
future RewardComposer
      |
      v
Gymnasium Env
```

Gymnasium will own framework-specific `spaces.Box`, NumPy conversion, episode reset/termination semantics, and adaptation between its array actions and this pure-Python `PolicyAction` contract.

Reward shaping, next-checkpoint progress authority, and policy-controlled braking remain separate milestones.