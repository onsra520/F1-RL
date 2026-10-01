# Track Asset Contract Design

**Status:** Written specification for review  
**Base branch:** `models/ppo-baseline`  
**Base implementation milestone:** WP1A verified

## 1. Goal

WP2 makes racing tracks portable, reproducible project assets instead of Python-only object construction.

A track asset must be able to answer:

- what world geometry and entities define this track;
- where the vehicle starts;
- whether two differently named files contain the same world geometry;
- whether the exact same track content and spawn were used later in an experiment;
- whether a manifest entry still matches the file on disk.

WP2 must preserve the existing deterministic sandbox and must not move experiment, training, reward, observation, or provenance configuration into track files.

## 2. Architectural boundary

Persistence is a new adapter above the sandbox:

```text
assets/tracks/*.json
        |
        v
f1rl/track_assets/
        |
        v
f1rl.sandbox.TrackWorld / WorldState
        |
        +--> learning/training
        +--> evaluation
        +--> API/editor later
```

The production dependency direction is:

```text
track_assets -> sandbox
learning     -> track_assets (when asset-backed scenarios are introduced)
sandbox      -X-> track_assets
```

`f1rl/track_assets` must not directly import:

- `f1rl.learning`;
- `f1rl.runtime`;
- Gymnasium;
- Stable-Baselines3;
- PyTorch;
- FastAPI;
- frontend/editor code.

The sandbox remains the deterministic domain core and does not know JSON, manifests, filesystems, or asset hashing.

## 3. Proposed module layout

```text
f1rl/track_assets/
├── __init__.py
├── model.py
├── loader.py
├── canonical.py
├── manifest.py
└── initial_state.py
```

Responsibilities:

- `model.py` — immutable asset DTOs such as `TrackAsset` and `SpawnPose`.
- `loader.py` — strict UTF-8 JSON parsing and construction of sandbox domain objects.
- `canonical.py` — semantic canonicalization and SHA-256 fingerprints.
- `manifest.py` — strict manifest loading and integrity verification.
- `initial_state.py` — deterministic mutable-state derivation from an immutable asset.

This is a small adapter package, not a generic serialization framework.

## 4. Track JSON schema v1

Canonical project track files use JSON.

Example shape:

```json
{
  "schema_version": 1,
  "track_id": "track-a",
  "spawn": {
    "position": [0.0, 1.0, 0.0],
    "orientation": [1.0, 0.0, 0.0, 0.0]
  },
  "surfaces": [],
  "obstacles": [],
  "boost_pads": [],
  "slow_zones": [],
  "checkpoints": [],
  "finish_lines": [],
  "score_pickups": []
}
```

All root keys are required in schema v1. Unknown keys are rejected.

`schema_version` must be integer `1`; booleans are not integers for validation purposes.

`track_id` is a non-empty string. It is an external asset identity and is not part of geometry/content fingerprints.

## 5. Spawn pose

`spawn.position` is the exact initial `VehicleState.position` reference-point position.

`spawn.orientation` is a quaternion encoded as:

```text
[w, x, y, z]
```

The loader constructs the existing sandbox `Quaternion`, which normalizes valid non-zero input.

The asset does not store velocity, angular velocity, grounded state, score, collected pickups, or active interaction state.

The loader does not silently snap spawn position to a surface or apply vehicle ride height. Authored spawn pose remains explicit data.

## 6. Surface representation

Each `surfaces` item maps directly to `PlanarSurfacePatch`:

```json
{
  "id": "surface-main",
  "origin": [0.0, 0.0, 20.0],
  "forward": [0.0, 0.0, 1.0],
  "normal": [0.0, 1.0, 0.0],
  "width": 10.0,
  "length": 40.0
}
```

The existing domain constructor remains authoritative for:

- non-empty IDs;
- unit `forward` and `normal` vectors;
- orthogonality;
- positive finite width/length.

The persistence adapter performs structural/type validation and then reuses domain validation rather than maintaining a second physics-validation rule set.

## 7. Volume representation

Volume-backed entities use `OrientedBoxVolume` encoded as:

```json
{
  "center": [0.0, 1.0, 30.0],
  "half_extents": [4.0, 2.0, 0.5],
  "orientation": [1.0, 0.0, 0.0, 0.0]
}
```

The existing `OrientedBoxVolume` constructor remains authoritative for positive half-extents and quaternion behavior.

## 8. Entity schemas

### Obstacle

```json
{
  "id": "obstacle-1",
  "volume": { "center": [0, 1, 10], "half_extents": [1, 1, 1], "orientation": [1, 0, 0, 0] }
}
```

### Boost pad

Adds:

```text
acceleration_bonus
```

and maps to the existing `BoostPad` domain type.

### Slow zone

Adds:

```text
drag_multiplier
```

and maps to the existing `SlowZone` domain type.

### Checkpoint

Adds:

```text
order
```

and maps to the existing `Checkpoint` domain type.

### Finish line

Contains only `id` plus `volume`.

### Score pickup

Adds:

```text
points
```

and maps to the existing `ScorePickup` domain type.

Every entity mapping is strict: required keys must be present and unknown keys are rejected.

## 9. Data deliberately forbidden from track files

Track assets do not contain experiment-owned or mutable runtime state.

Forbidden concerns include:

```text
VehicleParameters
SimulationConfig/fixed_dt/max_surface_distance
sensor/rangefinder configuration
observation scaling
reward composition or weights
episode max_steps
PPO or vector-environment configuration
training budget
training/evaluation split designation
runtime provenance
velocity
angular_velocity
grounded
score
collected pickup IDs
active checkpoint IDs
active finish-line IDs
```

This prevents track-dependent experimental confounds.

## 10. Strict JSON parsing

`load_track_asset(path)` reads UTF-8 JSON and rejects malformed or ambiguous input before domain construction.

The parser must reject:

- duplicate JSON object keys;
- `NaN`;
- `Infinity` / `-Infinity`;
- booleans where numeric values are expected;
- missing required keys;
- unknown keys;
- malformed vector/quaternion lengths;
- unsupported schema versions.

No hidden defaults fill omitted research-relevant or geometry-relevant fields.

JSON array order for entity collections is not semantically meaningful; IDs determine canonical ordering.

## 11. `TrackAsset` contract

Conceptual immutable DTO:

```python
@dataclass(frozen=True, slots=True)
class TrackAsset:
    schema_version: int
    track_id: str
    spawn: SpawnPose
    world: TrackWorld
    world_sha256: str
    content_sha256: str
```

`TrackWorld` continues to canonical-sort each entity category by `id` and enforce ID uniqueness across categories.

The hashes are calculated from canonical semantic payloads, never trusted from the source track JSON.

## 12. Dual fingerprint design

WP2 defines two separate SHA-256 fingerprints.

### `world_sha256`

Identifies world geometry/entities only.

It excludes:

- `track_id`;
- spawn pose;
- JSON whitespace;
- JSON object-key order;
- input entity-array order.

Purpose:

> detect the same circuit/world even when a file is copied, renamed, reformatted, reordered, or assigned another `track_id`/spawn.

This is the fingerprint WP3 can use for geometry leakage detection.

### `content_sha256`

Identifies:

```text
canonical world
+
canonical spawn pose
```

It still excludes `track_id` and source-file formatting.

Purpose:

> distinguish the exact semantic track+spawn content used by an experiment while allowing asset identity/name changes without changing the content identity.

WP6 can later record both fingerprints in provenance.

## 13. Hash-domain versioning

Raw JSON bytes are never hashed directly.

The canonical preimages are domain-separated and versioned.

Conceptually:

```text
world_sha256   = SHA256(canonical JSON of {"hash_schema": "f1rl-track-world-v1", ...world...})
content_sha256 = SHA256(canonical JSON of {"hash_schema": "f1rl-track-content-v1", ...world..., "spawn": ...})
```

The hash schema is distinct from the source-file `schema_version`.

A future file schema may preserve the same semantic hash if it resolves to the same canonical v1 world/content representation. A future change to fingerprint semantics requires a new hash-schema identifier.

## 14. Canonicalization rules

Before hashing:

1. all sandbox/entity DTOs are already validated;
2. entity arrays are sorted by `id`;
3. mapping keys are serialized with deterministic sorted ordering;
4. vectors serialize in fixed component order `[x, y, z]`;
5. quaternions serialize in fixed component order `[w, x, y, z]`;
6. quaternions are normalized;
7. equivalent quaternion signs `q` and `-q` canonicalize to one representation;
8. quaternion sign is selected so the first non-zero component in `(w, x, y, z)` is positive;
9. negative zero canonicalizes to positive zero;
10. non-finite numbers are impossible because they are rejected before hashing;
11. canonical JSON uses UTF-8, no insignificant whitespace, and stable key ordering.

Canonicalization is based on validated semantic objects, not the original textual spelling of numbers.

The implementation must include regression vectors that pin these rules so future refactors cannot silently change hashes.

## 15. Deterministic initial mutable state

A track file owns only spawn pose. Mutable initial state is derived canonically.

Conceptual API:

```python
build_initial_world_state(
    asset: TrackAsset,
    *,
    max_surface_distance: float,
) -> WorldState
```

The builder accepts the scalar `max_surface_distance` instead of importing `SimulationConfig`, preserving `track_assets -> sandbox` dependency direction.

Derived values are:

```text
position          = asset.spawn.position
orientation       = asset.spawn.orientation
velocity          = Vec3(0, 0, 0)
angular_velocity  = Vec3(0, 0, 0)
score             = 0
collected pickups = empty
active checkpoints = empty
active finish lines = empty
```

Initial `grounded` is derived as:

```python
asset.world.surface_at(
    asset.spawn.position,
    max_surface_distance,
) is not None
```

The builder validates `max_surface_distance` as a finite non-negative numeric value and rejects booleans.

The builder does not mutate or snap spawn pose.

This keeps initial observation grounded-state consistent with the same surface-query semantics used by simulation while preventing authored tracks from choosing arbitrary mutable state.

## 16. Race validity boundary

The generic asset loader may construct a valid `TrackWorld` that is not a valid racing episode world. For example, generic domain data can contain zero finish lines even though the RL race route requires exactly one finish.

`f1rl.track_assets` therefore does **not** import `f1rl.learning.episode.RaceRoute`.

Race-ready validation happens at the learning/integration layer by calling the existing:

```python
RaceRoute.from_track(asset.world)
```

That existing contract remains authoritative for:

- exactly one finish line;
- checkpoint orders contiguous from zero;
- checkpoint ID uniqueness;
- finish/checkpoint ID separation;
- zero checkpoints being valid.

WP2 corpus integration tests must prove every candidate research track is race-ready through that existing API without creating a production dependency from `track_assets` to `learning`.

## 17. Manifest schema v1

Canonical manifest path:

```text
assets/tracks/manifest.json
```

Example:

```json
{
  "schema_version": 1,
  "tracks": [
    {
      "track_id": "track-a",
      "path": "training/track-a.json",
      "world_sha256": "<64 lowercase hex chars>",
      "content_sha256": "<64 lowercase hex chars>"
    }
  ]
}
```

Manifest v1 is an asset registry, not the scientific train/held-out split contract.

Directory names such as `training/` and `evaluation/` remain organizational until WP3 freezes split semantics.

## 18. Manifest path rules

Manifest track paths:

- must be relative paths;
- must not be empty;
- must not resolve outside the manifest directory;
- must point to files;
- resolve relative to the manifest file itself.

Absolute paths and traversal outside `assets/tracks` via entries such as `../..` are rejected.

This preserves repository portability and prevents the manifest from becoming a generic filesystem indirection layer.

## 19. Manifest integrity checks

`load_track_manifest(path)` validates structural correctness and verifies each referenced file.

It must require:

- unique manifest `track_id` values;
- unique normalized relative paths;
- valid lowercase 64-character SHA-256 values;
- referenced file exists;
- loaded asset `track_id` matches manifest entry;
- loaded `world_sha256` matches manifest entry;
- loaded `content_sha256` matches manifest entry.

Any mismatch is an error; hashes are never silently regenerated into the checked-in manifest during loading.

A separate explicit developer utility may later generate/update manifest entries, but transparent mutation is out of scope for WP2.

## 20. Public API

Expected public surface:

```python
from f1rl.track_assets import (
    SpawnPose,
    TrackAsset,
    TrackManifest,
    TrackManifestEntry,
    load_track_asset,
    load_track_manifest,
    build_initial_world_state,
)
```

Exact internal helper names may differ in the implementation plan, but production consumers must not need to know JSON parsing/canonicalization internals.

## 21. Scenario construction relationship

WP2 does not redesign `F1RLEnvFactory`.

Later integration resolves the experiment-owned configuration and combines it with the asset:

```python
asset = load_track_asset(track_path)
initial_state = build_initial_world_state(
    asset,
    max_surface_distance=experiment.environment.simulation.max_surface_distance,
)

factory = F1RLEnvFactory(
    track=asset.world,
    initial_state=initial_state,
    vehicle_parameters=...,
    simulation_config=...,
    observation_encoder=...,
    reward_composer=...,
    episode_config=...,
)
```

The asset contributes only world/spawn information.

## 22. WP2A — asset infrastructure

WP2A implements:

- immutable asset DTOs;
- strict JSON loader;
- sandbox object construction;
- canonicalization;
- dual hashes;
- manifest model/loader/integrity verification;
- deterministic initial-state builder;
- focused documentation and tests.

WP2A does not invent actual research-track geometry merely to exercise the loader. Tests use dedicated test fixtures.

## 23. WP2B — candidate track corpus

WP2B authors the actual project track assets needed to establish representative geometry scale.

Expected repository shape:

```text
assets/tracks/
├── manifest.json
├── training/
│   └── *.json
├── evaluation/
│   └── *.json
└── demo/
    └── *.json
```

The current research direction is approximately four candidate training-side tracks plus two candidate evaluation-side tracks, but WP2 does not scientifically freeze that split.

WP2B must ensure:

- every checked-in track loads through the production loader;
- manifest hashes match;
- every research candidate is race-ready through `RaceRoute.from_track`;
- authored dimensions establish enough evidence for WP1B to choose sensor ranges, observation scales, reward weights, and episode horizon.

## 24. Relationship with WP1B

WP1A deliberately left vehicle/simulation/sensor/reward/episode numeric values unresolved.

After WP2B establishes real track scale, WP1B can justify and freeze:

- vehicle numeric parameters;
- fixed simulation timestep and surface-query distance;
- sensor count/directions/ranges;
- observation normalization scales;
- reward weights;
- episode horizon.

WP2 must not set those values inside track data.

## 25. Relationship with WP3

WP3 owns scientific split and leakage rules.

It may use WP2 fingerprints to enforce:

```text
training track IDs ∩ held-out track IDs = empty
training world hashes ∩ held-out world hashes = empty
```

Using `world_sha256` rather than a hash containing `track_id` prevents a copied/renamed circuit from bypassing the leakage guard.

WP3 also verifies shared non-track experiment configuration. WP2 does not.

## 26. Relationship with WP6

WP6 runtime provenance can record:

```text
track_id
world_sha256
content_sha256
```

WP2 computes stable semantic fingerprints; WP6 records the actual inputs used by a run.

## 27. Error behavior

Public load/build APIs fail fast with `ValueError` for invalid schema/content and propagate appropriate filesystem errors for missing/unreadable paths unless the implementation plan defines one narrow project-owned exception type.

Error messages should identify the failing field/path sufficiently for authored track debugging.

WP2 does not introduce a broad error hierarchy.

## 28. Testing requirements

WP2 implementation follows TDD and keeps mocks/test doubles in dedicated test code only.

Focused tests must cover at least:

### Parser/schema

- valid complete track loads;
- duplicate JSON keys rejected;
- unknown root/nested keys rejected;
- missing required keys rejected;
- unsupported schema rejected;
- bool-as-number rejected;
- `NaN`/infinity rejected;
- wrong vector/quaternion lengths rejected.

### Domain construction

- all seven `TrackWorld` categories round-trip into correct domain types;
- domain ID uniqueness remains authoritative;
- invalid unit/orthogonality/extents/entity numeric rules propagate as validation failures.

### Canonicalization/hashing

- source whitespace does not change hashes;
- JSON object-key order does not change hashes;
- input entity-array order does not change hashes;
- equivalent `q` and `-q` do not change hashes;
- negative zero does not change hashes;
- changing only `track_id` changes neither hash;
- changing only spawn changes `content_sha256` but not `world_sha256`;
- changing geometry/entity content changes both hashes;
- fixed regression vectors pin known hashes for canonical fixtures.

### Initial state

- velocity/angular velocity are zero;
- score and all mutable sets are empty;
- spawn pose is preserved;
- grounded derives from surface query;
- no surface yields `grounded=False`;
- invalid `max_surface_distance` rejected.

### Manifest

- relative paths resolve from manifest directory;
- absolute/out-of-root paths rejected;
- duplicate IDs/paths rejected;
- malformed hashes rejected;
- ID mismatch detected;
- world/content hash tampering detected.

### Integration/boundaries

- candidate research tracks are race-ready via `RaceRoute.from_track`;
- `sandbox` does not import `track_assets`;
- `track_assets` does not import `learning`, `runtime`, SB3, Torch, Gymnasium, or API/frontend code;
- existing SB3-owner invariant remains exactly two production files;
- existing production Torch-import invariant remains zero;
- existing suite remains green.

## 29. Non-goals

WP2 does not implement:

- track editor UI;
- arbitrary mesh/curve/spline track formats;
- binary asset format;
- schema migration engine;
- database storage;
- network loading;
- experiment train/held-out split logic;
- automatic manifest mutation;
- sensor/reward/physics tuning;
- evaluation metrics;
- training or policy evaluation;
- runtime provenance capture.

## 30. Acceptance criteria

WP2 is complete when:

1. track assets are strict portable JSON files;
2. production loader creates validated existing sandbox domain objects;
3. no mutable initial runtime state is authorable except spawn pose;
4. canonical initial state is derived deterministically;
5. semantic `world_sha256` and `content_sha256` are stable and meaningfully distinct;
6. renamed/reformatted/reordered identical worlds retain the same world fingerprint;
7. manifest integrity detects file/content drift;
8. real candidate track assets exist and load through the production path;
9. research candidate tracks pass existing race-route validity;
10. sandbox and RL dependency boundaries remain intact;
11. WP2 supplies enough real geometry scale for WP1B without silently freezing WP3 split semantics.

## 31. Planned sequence after WP2 approval

```text
WP2 written spec
→ WP2 implementation plan
→ WP2A TDD implementation
→ WP2A verification/review
→ WP2B candidate track corpus
→ WP2 full verification/review
→ WP1B environment numeric freeze
→ WP3 train/held-out split contract
```

No WP1B or WP3 implementation begins inside WP2.