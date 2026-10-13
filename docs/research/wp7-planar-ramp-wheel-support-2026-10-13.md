# WP7 2B — Prospective single-planar-ramp wheel and body support (2026-10-13)

**Status:** Draft research implementation; NOT a complete ramp/airborne physics solution. No scientific freeze.

## What this change adds

- **Explicit wheel reference points** `WheelContactPoints(front_left, front_right, rear_left, rear_right)`: four distinct, correctly ordered body-local tire-ground points. This does not infer official axle track, tires or wheelbase dimensions.
- **Typed support policy** `PlanarRampPolicy` with an explicit positive tire-to-plane tolerance and subdivision budget. Values in synthetic tests are **not** official car specifications.
- `certify_single_planar_ramp(...)` verifies **all four wheel contact points** remain inside *one* upward-facing planar patch and within its plane-distance tolerance, simultaneously with the **full projected 3D body box** fitting within that same finite road rectangle.
- For motion between physics states, the test assumes **linear translation + shortest-arc quaternion interpolation (SLERP)**, matching the existing prospective swept obstacle/road kernels. It uses component-specific conservative bounds along the patch's right, forward and normal axes and recursively subdivides only if necessary.
- Clear is only possible when every point can be **certified throughout the whole interval**. If a point lies outside, report `outside`; if precision is insufficient, `uncertain`; if the state is airborne or patch faces sideways/down, `unsupported`.
- The API is **opt-in** through `step_world → Simulation → F1RLEnv → F1RLEnvFactory`. When no `PlanarRampPolicy` is provided, the existing **flat-road union v1 verifier** remains unchanged. Original PPO 200k runs use neither prospective body physics mode.
- A prospectively active ramp policy is **strictly limited to exactly one surface patch**. Multiple patches, varying normals across ramp transitions, elevated decks, takeoff and landing remain **fail-closed**. These cases cannot accidentally fall back to centre-point validity.
- `TrainingScenarioSet` includes the ramp policy in shared semantic equality. `TrainingRunner` retains its **full-body training block**, including ramp mode. The old 200k weights are NOT treated as full-body or ramp-certified.

## What is NOT yet proved (required for project choice 1B/2B)

1. No approved tire radius, contact patch area, axle position, chassis centre offset, rigid body dimension, suspension, tire loads or friction. Synthetic tests cannot substitute for accepted vehicle specification.
2. No **multi-patch ramp/flat transition** support. For a physically valid smooth connection, each wheel may contact a different surface. Current v1 rejects it rather than fabricating support.
3. No **airborne/takeoff/landing** event protocol or vertical swept hull/road-surface continuous intersection. Legacy integrator's grounded flag and sampled centre contact are not an independently validated 3D landing rule.
4. No full-body legal route/corner-cutting threshold approved independently of the archived models. Four wheel points within a patch do not imply tire footprint, traction or road stability.
5. No approved scientific protocol/provenance, no independent PPO retrain and no HELD-OUT evaluation. Ramp is part of TRAIN, but this stage uses **synthetic tests only**, not archived or HELD-OUT track rollouts.
6. The kernel is intentionally conservative. At surface boundaries it may return `uncertain` even if a more powerful union-of-surface solver could prove the motion safe. Never silently convert this to a successful rollout.

## Verification contract

- Synthetic slope tests (0°, 15°, 30°), body overhang, one-tire off-road, elevated "false grounded" reference pose, mid-step yaw sweep, resolution exhaustion, unsupported airborne/vertical surface and invalid tolerance.
- Full prospective world-step integration for a single synthetic planar ramp. Verify original full-body flat union correctly blocks a sloped ramp **without** explicitly enabled ramp policy; multi-surface seam and missing geometry must still error.
- Factory/training tests verify ramp policy mismatches fail before scientific training or artifact creation.
- CI on **Python 3.12 Windows + Ubuntu**: targeted physics tests, full regression, compileall, pip check. Evaluate CI on the **latest branch HEAD**, not old branch snapshots.

**Decision:** WP7 full-body Ramp & Airborne remains **BLOCKED** until multi-surface tire contact and physically specified takeoff/landing are implemented, independently reviewed and re-audited. No full-body PPO training, no scientific freeze, no WP8, HELD-OUT sealed.
