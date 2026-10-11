# WP7 C2.1 recovery — authored navigation track content

Date: 2026-10-11. Worktree: `D:/Projects/.worktrees/F1-RL/wp7-rebuild`. This is a **new, independently implemented reconstruction**; old C2 commits, model weights and evidence are unavailable.

## Implemented boundary

Six canonical assets use track **schema_version=2** with typed, explicit authored navigation:

```json
{"navigation": {
  "points": [[0, 0, 4], [0, 0, 35], [0, 0, 75], [0, 0, 112]],
  "surface_ids": ["road-main", "road-main", "road-main"],
  "checkpoint_point_indices": [1, 2]
}}
```

- Each surface ID maps a single waypoint segment to an existing planar patch. Validated at segment endpoints and interior samples, with at most 1e-4 distance from patch plane.
- Navigation begins at spawn, ends at finish center, and each checkpoint index maps to the checkpoint of matching ascending race order.
- Nonfinite coordinates, adjacent duplicate waypoints, gaps/mismatched surfaces, extra keys, invalid indices and off-surface points reject at load time.
- Schema-1 fixtures retain their old parser and `f1rl-track-content-v1` hash; they have no authored navigation.
- Schema-2 content hash uses `f1rl-track-content-v2`, including canonical waypoint coordinates, surface IDs and checkpoint indices.
- `f1rl-track-world-v1` hash identity is unchanged. No `TrackWorld` surface, obstacle, checkpoint, finish, modifier, spawn or physics value changed.
- The manifest remains schema-1, containing six entries with their newly calculated content hashes.

## World fingerprint hard gate

| Track | SHA256 (unchanged, entire value) |
| --- | --- |
| straight-sprint-v1 | 685f273f62bdc35399d73be5db922198022ad2b549623924dbeb8171d4a4088b |
| slalom-v1 | 6071829f2eb790ada04c5e59b22efb205e3a20d59fbe670e68fb78ed3eb304e0 |
| right-turn-v1 | d79133bc9a0599f9b056c7eb78658446631f564f038bc4afa91f3325fd4f9fd2 |
| ramp-v1 | 775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978 |
| left-turn-v1 | 4a613e8c851a9940395f81c54857c9b94f903cb893e29401b0457bebe8a10adf |
| offset-slalom-v1 | 32d903e81880ea290c840fd210ca2c43579004135acf36d58c49bd73d6c91346 |

## Separation from later C2 steps

**C2.1 does not alter any learned model evaluation or runtime navigation behavior**. `F1RLEnv` still targets direct checkpoint centers and `TargetProgressReward` still uses target distance. C2.2 must wire route progression, route-local observation, monotone high-water credit and off-track termination/reward into the simulator and training factory. `ExperimentDefinition` and scientific protocol status remain draft.

Authoring route geometry is a static input contract, not a claim of actual pilot 4/4 performance. No held-out learned-policy result has been inspected. No PPO training, final replicate use or WP8 work performed.

## Validation

- RED phase: newly introduced schema-2 test failed on WP6 `schema_version=1` and absent navigation type.
- Canonical migration computed all six v2 content fingerprints, checked all six world hashes against the original manifest.
- Targeted track-asset suite: **90 passed, 1 skipped**.
- Whole reconstructed branch suite: **1094 passed, 2 skipped** before commit.
