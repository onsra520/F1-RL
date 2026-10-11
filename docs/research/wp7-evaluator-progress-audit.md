# WP7 recovery: evaluator high-water metric contract (2026-10-11)

## Failure reproduced

The WP6 `evaluate_episode` implementation used `_normalized_route_progress` over straight chords joining the start, checkpoint centers and finish. The WP7 reconstruction no longer rewards direct checkpoint-distance progress; it rewards an authored 3D polyline's nondecreasing per-episode high-water value.

The new RED regression reproduced an incorrect result after a policy moved forward and then reversed: the evaluator returned **0.0185185185** from the final position while the episode's certified high-water telemetry was **5.0 / 108.0 = 0.0462962963**.

## Fix and evidence gate

When `F1RLEnvFactory.navigation` is provided, the evaluator:
- Requires the authoritative `route_highwater_meters` on every step.
- Rejects non-finite, missing, decreasing, negative and checkpoint-ineligible reported progress. Bounds are the authored route length and next *uncompleted* checkpoint.
- Computes `final_normalized_route_progress` as maximum attained meters divided by authored polyline length, or 1.0 on confirmed finish.
- Preserves the WP6 legacy chord estimator for factories without authored navigation, so existing fixtures and tests retain their intended semantics.
- Does **not** evaluate a learned policy or inspect held-out training outcomes during this audit.

The new regression uses scripted diagnostic rollout telemetry, not trained policies. This metric change requires protocol/version review before freezing WP7; older WP6 summaries and new WP7 summaries cannot be compared as though they used identical progress estimators.

## Separate static route findings

A preliminary TRAIN-only geometric sampling found:
- Straight: 108m authored polyline, no obstacles.
- Ramp: ~149.58m authored polyline, no obstacles.
- Right turn: 130m authored polyline with one abrupt **90-degree** waypoint junction.
- Slalom (initial pre-repair baseline at 2026-10-11): ~127.96m authored polyline, **~1.15m minimum planar clearance**, and **~75-degree** and **~62-degree** junctions near the finish. **Superseded** by the later TRAIN-only route repair in `wp7-train-dynamic-route-audit-2026-10-11.md` (new sampled clearance ~1.94m, maximum corner ~43.4°; world geometry unchanged).

These are **risk observations**, not proof of physical infeasibility or successful pilot performance. The world SHA256s remain unchanged. A durable static audit validator and physical driveability review remain necessary before training.
