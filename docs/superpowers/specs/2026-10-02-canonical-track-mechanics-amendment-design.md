# Canonical Track Mechanics Amendment Design

**Status:** Written specification for review
**Branch:** `research/track-mechanics-amendment`
**Base HEAD:** `7dc91cf900ed51ca2861d7b64bccb5998db4f0ec`

## 1. Goal

This amendment makes boost and slow-zone mechanics part of the actual canonical six-track research corpus before WP4 telemetry and before any pilot/baseline training.

The engine already supports:
- vehicle dynamics and velocity;
- gravity and grounded/airborne state;
- physical obstacles;
- boost pads;
- slow zones;
- checkpoints and finish lines.

However, the current six canonical track assets contain obstacles and elevation but zero boost pads and zero slow zones. The purpose of this amendment is to remove that experiment-design gap without changing the core research question.

## 2. Research question remains unchanged

The experiment remains:

> Generalization of a PPO waypoint-guided navigation/control policy to held-out track geometries.

Boost and slow-zone mechanics must therefore be exposed during training before they appear in held-out evaluation. Held-out tracks must not introduce a new mechanic class that the policy never encountered during training.

This amendment changes canonical track content, not the research objective.
## 3. Existing protocol boundaries

### WP1 / WP1B remain authoritative

This amendment does not change:
- `VehicleParameters`;
- `SimulationConfig`;
- rangefinder layout or range;
- observation features or normalization;
- action semantics;
- reward composition or weights;
- episode horizon;
- experiment draft/frozen lifecycle.

The canonical WP1B values remain exactly unchanged.

### WP2 ownership

Boost pads and slow zones are track-owned entities. Therefore their placement and strengths belong in canonical track JSON assets, not in the experiment YAML.

### WP3 ownership

The train/held-out split remains 4 + 2 with the same IDs. WP3 leakage guards continue to use `world_sha256`, so any modified track asset must be rehashed and revalidated.

## 4. Canonical mechanic-exposure design

The six-track corpus becomes:

| Track | Partition | Mechanic role |
| --- | --- | --- |
| `straight-sprint-v1` | training | pure flat longitudinal dynamics |
| `slalom-v1` | training | obstacle avoidance + boost exposure |
| `right-turn-v1` | training | right-turn geometry + slow-zone exposure |
| `ramp-v1` | training | pure elevation/gravity dynamics |
| `left-turn-v1` | held-out | mirrored turn geometry + known slow-zone mechanic |
| `offset-slalom-v1` | held-out | altered slalom geometry + known boost mechanic |

Only four assets change:
- `slalom-v1`;
- `right-turn-v1`;
- `left-turn-v1`;
- `offset-slalom-v1`.

`straight-sprint-v1` and `ramp-v1` remain byte-identical and must retain their current semantic hashes.
## 5. Boost design

The training and held-out slalom pair each receive one boost pad.

Canonical strength:

```text
acceleration_bonus = 3.0 m/s^2
```

The boost zone must:
- lie entirely over a valid flat road surface;
- be encountered before the main obstacle field;
- be long enough for at least several 0.1 s simulation ticks at realistic approach speeds;
- not overlap spawn, checkpoint volumes, finish volume, or obstacles;
- not sit on a surface transition;
- be reachable while grounded.

The training slalom and held-out offset-slalom should have equivalent mechanic exposure, not necessarily byte-identical zone coordinates. Their geometry remains different.

The intent is to make higher-speed obstacle control observable while preserving the existing `max_ground_speed = 15.0 m/s` cap.

## 5A. Exact canonical volumes

The implementation must use these exact entities.

### slalom-v1

```json
{
  "id": "boost-0",
  "volume": {
    "center": [0, 1, 36],
    "half_extents": [7, 1, 4],
    "orientation": [1, 0, 0, 0]
  },
  "acceleration_bonus": 3.0
}
```

This spans z=32..40 on the 14 m wide road. It starts after checkpoint-0 (which ends at z=31.5) and ends before obstacle-0 (which begins at z=43).

### offset-slalom-v1

```json
{
  "id": "boost-0",
  "volume": {
    "center": [0, 1, 36],
    "half_extents": [8, 1, 4],
    "orientation": [1, 0, 0, 0]
  },
  "acceleration_bonus": 3.0
}
```

This spans z=32..40 on the 16 m wide road. It starts after checkpoint-0 (which ends at z=29.5) and ends before obstacle-0 (which begins at z=43).

### right-turn-v1 and left-turn-v1

Both tracks receive the same slow-zone volume:

```json
{
  "id": "slow-0",
  "volume": {
    "center": [0, 1, 50],
    "half_extents": [6, 1, 8],
    "orientation": [1, 0, 0, 0]
  },
  "drag_multiplier": 2.0
}
```

This spans z=42..58 across the full 12 m entry road. It begins after checkpoint-0 (which ends at z=36) and remains 12 m clear of the entry-road end/turn transition at z=70.

These exact volumes are part of the amended canonical corpus. Changing their coordinates, extents, orientation, IDs, or modifier values is a protocol-content change that requires explicit review.

## 6. Slow-zone design

The right-turn and left-turn pair each receive one slow zone.

Canonical strength:

```text
drag_multiplier = 2.0
```

With WP1B `rolling_drag = 0.4`, full-throttle longitudinal equilibrium inside the zone is approximately:

```text
6.0 - (0.4 * 2.0 * v) = 0
v ~= 7.5 m/s
```

The zone must:
- lie on the straight entry section before the 90-degree turn;
- use mirrored/equivalent placement across right-turn and left-turn tracks;
- not overlap spawn, checkpoint, finish, or surface transitions;
- be reachable while grounded;
- be long enough to measurably affect speed without acting as a hard stop.

The intent is controlled speed-management pressure before a turn, not a braking substitute.
## 7. Why the other two tracks remain unchanged

`straight-sprint-v1` remains a clean flat-road control track.

`ramp-v1` remains a clean elevation/gravity control track.

Keeping these two tracks free of modifiers preserves interpretability. If every track contained boost or slow effects, failures would be harder to attribute between geometry, modifier exposure, gravity, and obstacle interaction.

## 8. Physics semantics

The implementation already applies track modifiers from the vehicle's current query at the start of a simulation tick.

Therefore:
- entering a zone at the end of tick N does not affect tick N;
- the modifier applies beginning on tick N+1 if the accepted vehicle state is inside the zone;
- boost and slow modifiers are ignored when there is no valid ground contact;
- canonical zones must be placed so they can be exercised under normal grounded traversal.

The amendment does not change these physics semantics.

## 9. Scientific exposure rule

Mechanic classes in held-out tracks must be a subset of mechanic classes exposed by training tracks.

For canonical protocol v1:

```text
training mechanics:
  obstacle
  boost
  slow
  elevation

held-out mechanics:
  obstacle
  boost
  slow
```

Held-out geometry may differ, but held-out tracks must not introduce a new dynamics mechanic class absent from training.

This preserves the intended interpretation of the experiment as geometry generalization rather than unseen-mechanic generalization.
## 10. Hash and manifest consequences

Adding a boost pad or slow zone changes canonical world content.

Therefore for each modified track:
- `world_sha256` must change;
- `content_sha256` must change;
- the corresponding manifest entry must be updated.

For `straight-sprint-v1` and `ramp-v1`, the hashes are pinned and must remain exactly:

```text
straight-sprint-v1
  world_sha256   685f273f62bdc35399d73be5db922198022ad2b549623924dbeb8171d4a4088b
  content_sha256 84adafbebb491af7bfbdaa9ef91e0c741023af2c5910c375cb3ce8d5f4e62070

ramp-v1
  world_sha256   775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978
  content_sha256 0fbf4b4633926944759c1972ecc15c6011785dc4d37d46f9d62c4c7d3a12c9b4
```

No manifest IDs or paths change.

WP3 split IDs remain unchanged.

## 11. Required canonical-asset validation

The amended corpus must prove:
- exactly one boost pad exists in each slalom pair member;
- exactly one slow zone exists in each turn pair member;
- no boost/slow zone exists in straight-sprint or ramp;
- canonical boost strength is exactly `3.0`;
- canonical slow multiplier is exactly `2.0`;
- every zone center lies over a valid track surface;
- every zone is reachable with a grounded vehicle;
- modifier volumes do not overlap obstacle volumes;
- modifier volumes do not overlap finish lines;
- modifier placement does not alter checkpoint ordering or route validity.

Where appropriate, overlap checks use the project geometry/query APIs rather than duplicating geometry math in tests.
## 12. Dynamics proof

The amendment must include integration tests against the actual canonical assets, not only schema assertions.

### Boost proof

For a grounded vehicle state inside the canonical boost zone:
- step once with the canonical track;
- step the same state/control against an equivalent track with the boost removed;
- the boosted case must produce greater longitudinal acceleration or post-step speed.

### Slow-zone proof

For a grounded moving vehicle state inside the canonical slow zone:
- step once with the canonical track;
- step the same state/control against an equivalent track with the slow zone removed;
- the slowed case must produce lower post-step speed.

These tests verify that authored canonical assets actually exercise the existing physics mechanics.

## 13. WP3 compatibility revalidation

After rehashing, rerun all existing WP3 guarantees:
- 4 training IDs and 2 held-out IDs unchanged;
- every manifest entry resolves;
- all six selected `world_sha256` values remain globally unique;
- no cross-partition geometry leakage;
- all six factories still share the frozen WP1B non-track contract;
- canonical initial-state fairness remains unchanged;
- all six race routes remain valid.

The amendment must not weaken any WP3 leakage rule.
## 14. WP4 relationship

WP4 remains blocked until this amendment is complete.

After the amended canonical corpus exists, WP4 telemetry may meaningfully expose:

```python
info["telemetry"] = {
    "vehicle_position": ...,
    "vehicle_velocity": ...,
    "vehicle_speed": ...,
    "grounded": ...,
    "obstacle_hit_ids": ...,
    "applied_boost_pad_ids": ...,
    "applied_slow_zone_ids": ...,
}
```

WP4 must still keep all evaluation telemetry out of policy observation.

This amendment does not implement telemetry.

## 15. Files expected to change

Primary asset changes:

```text
assets/tracks/training/slalom-v1.json
assets/tracks/training/right-turn-v1.json
assets/tracks/evaluation/left-turn-v1.json
assets/tracks/evaluation/offset-slalom-v1.json
assets/tracks/manifest.json
```

Expected test/documentation changes:

```text
tests/integration/test_track_asset_corpus.py
tests/integration/test_experiment_track_split_integration.py
tests/integration/...canonical mechanics proof...
assets/tracks/README.md
configs/experiments/README.md
```

No production physics implementation change is expected.
## 16. Non-goals

This amendment does not:
- change physics algorithms;
- change vehicle/simulation values;
- change PPO observation;
- change reward;
- add brake control;
- change the 4 + 2 split IDs;
- add new tracks;
- add telemetry;
- implement evaluator metrics;
- run pilot or baseline training;
- change training seeds or budget;
- redesign score-pickup mechanics.

## 17. Acceptance criteria

The amendment is complete when:
1. slalom and offset-slalom each contain one canonical boost pad with `acceleration_bonus = 3.0`;
2. right-turn and left-turn each contain one canonical slow zone with `drag_multiplier = 2.0`;
3. straight-sprint and ramp remain semantically unchanged;
4. canonical modifier zones lie on valid grounded road regions and avoid obstacle/finish conflicts;
5. integration tests prove boost and slow modifiers affect real canonical vehicle dynamics;
6. the four modified asset hashes and manifest entries are updated;
7. unchanged asset hashes remain unchanged;
8. WP3 4 + 2 membership remains unchanged;
9. WP3 world-hash leakage guards remain green;
10. WP1B non-track environment equality remains green;
11. observation/action/reward/race semantics remain unchanged;
12. the complete test suite and dependency boundaries remain green.

## 18. Sequence after approval

```text
mechanics amendment written spec
-> implementation plan
-> TDD canonical asset assertions
-> edit four track assets
-> recompute hashes + manifest
-> canonical dynamics proof
-> WP3 compatibility revalidation
-> full verification
-> return to WP4 Evaluation Telemetry
```
