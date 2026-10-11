# WP7 C2.3a — high-water route reward, training-free reconstruction

**Status:** Implementation candidate / verification, 2026-10-11. Previous historical implementation and its training evidence are unavailable; this version is new code.

## Responsibility split

- `AuthoredRoute.project()` and `advance_highwater()` supply *checkpoint-gated* route arc-length measurements. Reward cannot be earned beyond the next uncompleted checkpoint; off-path projections beyond a maximum allowable distance grant no new progress.
- `F1RLEnv` owns per-episode `_route_highwater` and resets it on every successful reset. It computes a candidate after stepping the simulation and processing ordered checkpoint events; commits the new high-water only on successful step finalization. Reward/observation errors leave the episode requiring reset.
- The maximum permitted lateral offset is the minimum half-width of the track's physical road surfaces. This is a reconstruction choice to independently review before final pilot.
- `RewardTransition.route_progress_delta_meters` is optional, finite and non-negative. Legacy 5-argument transitions remain fully compatible.
- `RouteProgressReward(weight_per_meter)` uses exactly `route_progress_delta_meters * weight_per_meter`. It rejects missing route evidence; no-nav environments configured with this reward term fail at construction.
- Guided `step()` additionally reports `info["route_highwater_meters"]` for diagnostics. Legacy telemetry and spaces are unchanged. No direct-target distance shaping is used *when the new term is the sole progress term*; old `TargetProgressReward` remains available for existing clients.

## Audit boundaries and future gates

This is **not** the complete C2 protocol. There is not yet an off-track penalty or failure criterion, nor a canonical `ExperimentDefinition -> F1RLEnvFactory` builder using the new semantics. The canonical YAML still declares the old draft v1 semantics, and there is no new training/pilot or held-out policy evaluation.

The high-water mechanism is not certified robust to all hypothetical shortcut/teleport geometries and its lateral-distance rule remains to be reviewed. Do not freeze v1 scientific protocol based on these unit tests.

## Verification

- RED test coverage added for missing reward class, missing/invalid route delta, wrong configs, and guided environment reward lifecycle.
- Focused reward/Gym test suite passed.
- Full suite: **1125 passed, 2 skipped in 31.16s**.
