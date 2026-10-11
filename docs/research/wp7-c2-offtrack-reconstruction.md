# WP7 C2.3b — bounded off-track penalty / failed-episode gate

**Reconstruction:** October 11, 2026. This is newly authored source code. Old WP7 model weights, proof artifacts, and identical historical algorithms were not recovered.

## Contract

- `OffTrackConfig(lateral_limit_meters, max_consecutive_steps)` requires **explicit** positive corridor and consecutive-step grace period. No hidden scientific defaults.
- The authored polyline's `horizontal_distance(position)` computes minimum XZ planar distance over *all* route segments. This deliberately ignores vehicle altitude during ramp flight, so a valid ballistic launch is not incorrectly considered off-track from vertical separation. All-segment corridor check is distinct from checkpoint-gated high-water reward.
- `OffTrackState.advance()` increments only for consecutive steps with distance **strictly greater** than the configured limit, and resets as soon as the vehicle re-enters the corridor.
- `OffTrackPenalty(penalty)` returns a per-step negative RL reward when `RewardTransition.off_track=True`; missing evidence is an error. Legacy transition constructors and rewards continue to work unchanged.
- A new `F1RLEnvFactory.offtrack_config` is propagated to each instance; with a configured monitor and authored navigation, threshold-strike exhaustion marks the episode **`truncated=True, terminated=False, finished=False`** and requires reset. Success on the same step takes precedence.
- Per-step `info` adds `offtrack_lateral_meters`, `offtrack_consecutive_steps` and `truncation_reason` for explicitly configured environments only. Legacy telemetry does not change.
- Multi-track `TrainingScenarioSet` requires shared off-track rules but **does not** require identical navigation routes. Different authored route arrays are track-specific; a TDD regression covers this.
- All changes are opt-in. Existing v1 configs instantiate no off-track monitor and retain established WP6 behavior.

## Research caveats

These thresholds are *candidate design choices*, not scientifically frozen parameters. The horizontal corridor does not, by itself, prove absence of shortcut exploits. Canonical builder/semantic identity and training-only pilot still need independent review and test. No PPO training, held-out policy evaluation, final seeds, frozen protocol or WP8 execution performed.

## Verification

- New red-phase tests initially failed to import the missing off-track module.
- New multi-track scenario contract test initially failed due to erroneously shared `navigation`, and passed after correction.
- Focused tests: **101 passed** before additional multi-track regression.
- Full suite after multi-track correction: **1138 passed, 2 skipped in 30.58s**. `compileall` and `uv pip check` pass.
