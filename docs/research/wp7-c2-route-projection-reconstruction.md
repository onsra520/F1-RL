# WP7 C2.2a — checkpoint-gated authored route projection

**Status:** Unit-level recovery, NOT wired into Gym or training. Reconstructed 2026-10-11 after losing the previous worktree.

`f1rl/learning/route_navigation.py` implements deterministic 3D polyline geometry on the six version-2 authored navigation assets:

- `AuthoredRoute.project(position, next_checkpoint_index=...)` projects a vehicle's 3D position onto the nearest authorized route segment. Segments beyond the next *uncompleted* checkpoint are inaccessible, eliminating unearned forward credit from simply reaching the finish before passing checkpoints.
- Each projection reports `progress_meters`, `off_path_meters`, `segment_index`, `segment_fraction`, and `endpoint` (the authored endpoint of the projected segment).
- Stable tie-breaking uses earliest segment index if two segments are equally near. Ramp progress includes vertical distance, not only XZ distance.
- `AuthoredRoute.advance_highwater(previous_meters, projection, max_path_distance=...)` takes a high-water value owned by the runtime: never decreases on backward travel; does not grant credit when a vehicle is far off the selected route. The route has no learned states or mutable global cache.
- Existing direct-target/reward/runtime contracts are untouched until the separate integration gate. The route geometry is a candidate semantic building block, not evidence that the policy can learn slalom or turn.

**Tests:** `tests/learning/test_route_navigation_reconstruction.py` covers straight-line progress, corner endpoints, 3D ramp distance, checkpoint gating, skip-to-finish denial, monotonic high-water, deterministic tie-break and invalid inputs. This file is training-free and does not evaluate held-out models.

**Next:** C2.2b integrates authored endpoint guidance into `F1RLEnvFactory` / `F1RLEnv` and canonical builder; C2.3 restores high-water reward, off-track penalty/failure and telemetry; then run training-only pilot with new archived evidence. Do **not** freeze based on old WP7 reports.
