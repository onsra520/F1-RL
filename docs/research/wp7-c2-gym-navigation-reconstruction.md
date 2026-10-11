# WP7 C2.2b — optional authored endpoint guidance in Gym environment

**Recovery status:** implemented, training-free. Date 2026-10-11. Follow-up to C2.1 and C2.2a.

- `F1RLEnvFactory` accepts optional `navigation: AuthoredNavigation | None`; factory propagation is explicit. A missing value retains the WP6 checkpoint-target behavior.
- `F1RLEnv` verifies navigation geometry identity against spawn, finish, ordered checkpoints and surface IDs; incompatible track/route pairs fail before rollout.
- `F1RLEnv._route_target(vehicle, progress)` uses checkpoint-gated `AuthoredRoute.project(...).endpoint` for v2 guided factories, or the existing `RaceRoute.target_for` for legacy factories.
- Reset observation is based on the next authored segment endpoint. Per step, the pre-step target drives the current target-progress reward, while the post-step observation uses the endpoint for the accepted vehicle state and updated checkpoint progress. Observation and action tensor shapes and legacy info telemetry remain unchanged.
- No PPO training, held-out learned-policy outcomes, scientific seed use or WP8 execution is authorized or performed by this code.
- **Not yet C2-complete:** reward remains `TargetProgressReward`, not checkpoint-gated high-water route progress; off-track penalties/termination and canonical experiment semantic identities must be migrated in a separately tested gate. A canonical experiment runtime builder is still needed to explicitly provide the track's `navigation` to its training factory.

New integration tests in `tests/integration/test_authored_route_env_reconstruction.py` cover ramp endpoint vs checkpoint target, right-turn corner segment, post-step guided observation, wrong route-to-track pairing, constructor types, and completed-race target None. Legacy Gym unit/integration tests run alongside these.
