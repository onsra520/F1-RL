# Track Asset Contract Design

**Status:** Written specification for review  
**Branch:** `models/ppo-baseline`  
**Prerequisite:** WP1A verified

## 1. Goal

WP2 turns racing tracks from Python-only construction into portable, reproducible project assets.

A loaded asset must provide:

- validated `TrackWorld` geometry/entities;
- one explicit vehicle spawn pose;
- a stable fingerprint for world geometry/entities;
- a stable fingerprint for world + spawn content;
- manifest integrity checks that detect asset drift.

WP2 must not move experiment, physics, observation, reward, training, evaluation-split, or provenance configuration into track files.

## 2. Architecture

Persistence is an adapter above the sandbox:

```text
assets/tracks/*.json
        |
        v
f1rl/track_assets/
        |
        v
f1rl.sandbox TrackWorld / WorldState
        |
        +--> learning/training
        +--> evaluation
        +--> API/editor later
```

Production dependency direction:

```text
track_assets -> sandbox
learning     -> track_assets when asset-backed scenarios are introduced
sandbox      -X-> track_assets
```

`f1rl.track_assets` must not import `f1rl.learning`, `f1rl.runtime`, Gymnasium, Stable-Baselines3, PyTorch, FastAPI, or frontend/editor code.

The sandbox remains the deterministic domain core and never learns about JSON, manifests, filesystems, or hashing.

## 3. Module layout

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

- `model.py` — immutable `TrackAsset`, `SpawnPose`, manifest DTOs as appropriate;
- `loader.py` — strict UTF-8 JSON parsing and sandbox object construction;
- `canonical.py` — semantic canonicalization and SHA-256 fingerprints;
- `manifest.py` — strict manifest loading and referenced-file integrity verification;
- `initial_state.py` — deterministic mutable-state derivation from asset spawn.

This is a small project adapter, not a generic serialization framework.

## 4. Track JSON schema v1

Canonical project tracks use JSON:

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

All root keys are required. Unknown keys are rejected. `schema_version` must be integer `1`, with booleans rejected as integers. `track_id` must be a non-empty string.

`track_id` is asset identity only and is excluded from both semantic fingerprints.

## 5. Spawn contract

`spawn.position` is the exact initial `VehicleState.position` reference-point position.

`spawn.orientation` is a quaternion in `[w, x, y, z]` order. Loading constructs the existing sandbox `Quaternion`, which validates finiteness/non-zero norm and normalizes it.

The loader does not snap position to a surface or apply vehicle ride height. Authored pose remains explicit data.

Track JSON never stores velocity, angular velocity, grounded state, score, collected pickups, or active interaction state.

## 6. Geometry/entity representation

Each surface maps directly to `PlanarSurfacePatch`:

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

Volume-backed entities share:

```json
{
  "center": [0.0, 1.0, 30.0],
  "half_extents": [4.0, 2.0, 0.5],
  "orientation": [1.0, 0.0, 0.0, 0.0]
}
```

Entity-specific fields are:

- `Obstacle`: `id`, `volume`;
- `BoostPad`: `id`, `volume`, `acceleration_bonus`;
- `SlowZone`: `id`, `volume`, `drag_multiplier`;
- `Checkpoint`: `id`, `volume`, `order`;
- `FinishLine`: `id`, `volume`;
- `ScorePickup`: `id`, `volume`, `points`.

Every mapping is strict: all required keys present, no unknown keys.

The adapter validates structure/types, then existing sandbox constructors remain authoritative for domain semantics such as unit/orthogonal surface vectors, positive extents, entity numeric constraints, and world-wide ID uniqueness.

## 7. Forbidden track-file concerns

Track files must not contain:

```text
VehicleParameters
fixed_dt / max_surface_distance
sensor/rangefinder configuration
observation scaling
reward configuration
episode max_steps
PPO/vector-environment configuration
training budget
scientific train/held-out designation
runtime provenance
velocity / angular_velocity
grounded
score
collected pickup IDs
active checkpoint/finish IDs
```

Only world geometry/entities and spawn pose vary per asset.

## 8. Strict JSON parsing

`load_track_asset(path)` reads UTF-8 JSON and rejects ambiguous/malformed data before domain construction.

It must reject:

- duplicate keys at every JSON object nesting level;
- `NaN`, `Infinity`, and `-Infinity`;
- booleans where numeric values are expected;
- missing required keys;
- unknown keys;
- malformed vector/quaternion lengths;
- unsupported schema versions.

No omitted field receives a hidden geometry or runtime default.

Entity collection input order is non-semantic; IDs determine canonical order.

## 9. TrackAsset contract

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

Hashes are computed from validated semantic objects and never trusted from source track JSON.

## 10. Dual semantic fingerprints

### `world_sha256`

Represents world geometry/entities only.

Excluded:

- `track_id`;
- spawn pose;
- JSON whitespace;
- source object-key order;
- source entity-array order.

Purpose: identify the same circuit/world even if copied, renamed, reformatted, reordered, or assigned another spawn. WP3 can therefore detect geometry leakage even after track renaming.

### `content_sha256`

Represents canonical world + canonical spawn pose, while still excluding `track_id` and source formatting.

Purpose: identify the exact semantic world/spawn input later recorded by WP6 provenance.

Changing only spawn changes `content_sha256`, not `world_sha256`.

## 11. Hash-domain versioning

Raw JSON bytes are never hashed.

Canonical preimages are domain-separated:

```text
world_sha256:
  {"hash_schema":"f1rl-track-world-v1", ...canonical world...}

content_sha256:
  {"hash_schema":"f1rl-track-content-v1", ...canonical world..., "spawn":...}
```

Source-file `schema_version` and hash-schema version are separate concepts.

A later file schema may retain the same semantic fingerprints when it resolves to the same canonical representation. A change in fingerprint semantics requires a new hash-schema identifier.

## 12. Canonicalization rules

Before hashing:

1. all semantic objects have passed validation;
2. each entity category is sorted by `id`;
3. mapping keys serialize in deterministic sorted order;
4. vector component order is `[x, y, z]`;
5. quaternion component order is `[w, x, y, z]`;
6. quaternions are normalized;
7. equivalent `q` and `-q` canonicalize to one sign: the first non-zero component in `(w, x, y, z)` is positive;
8. negative zero canonicalizes to positive zero;
9. all continuous numeric fields canonicalize to finite floating-point values before JSON serialization, so semantically equal source values such as `1` and `1.0` hash identically;
10. discrete fields (`schema_version`, checkpoint `order`, pickup `points`) remain integers;
11. canonical JSON is UTF-8, stable-key ordered, and contains no insignificant whitespace.

Canonicalization uses validated semantic values, not original textual numeric spelling.

Tests must pin known hash regression vectors so refactors cannot silently alter fingerprint semantics.

## 13. Deterministic initial mutable state

Track data owns only spawn pose. Mutable initial state is derived:

```python
build_initial_world_state(
    asset: TrackAsset,
    *,
    max_surface_distance: float,
) -> WorldState
```

The function takes the scalar distance rather than importing `SimulationConfig`, preserving `track_assets -> sandbox` direction.

Derived state:

```text
position           = asset.spawn.position
orientation        = asset.spawn.orientation
velocity           = Vec3(0, 0, 0)
angular_velocity   = Vec3(0, 0, 0)
score              = 0
collected pickups  = empty
active checkpoints = empty
active finish lines = empty
```

Initial grounded state is:

```python
asset.world.surface_at(
    asset.spawn.position,
    max_surface_distance,
) is not None
```

`max_surface_distance` must be finite, non-negative, numeric, and not boolean.

The builder does not snap/mutate spawn pose. Grounded derivation exists so the initial observation agrees with the same surface-query semantics used by simulation.

## 14. Race-validity boundary

Generic `track_assets` may load a valid `TrackWorld` that is not a valid RL race course. Therefore production `track_assets` does not import `learning`.

Race-ready validation stays at the learning/integration layer using existing:

```python
RaceRoute.from_track(asset.world)
```

That contract remains authoritative for:

- exactly one finish line;
- checkpoint orders contiguous from zero;
- checkpoint ID uniqueness;
- finish/checkpoint ID separation;
- zero checkpoints being valid.

WP2 candidate-corpus integration tests call this existing API without creating `track_assets -> learning` production dependency.

## 15. Manifest schema v1

Canonical path:

```text
assets/tracks/manifest.json
```

Shape:

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

The manifest is an asset registry, not the scientific train/held-out split contract. Existing `training/`, `evaluation/`, and `demo/` directory names remain organizational until WP3 freezes research split semantics.

## 16. Manifest path/integrity rules

Manifest paths:

- are non-empty relative paths;
- resolve relative to the manifest directory;
- must remain inside that directory after normalization;
- must refer to files;
- cannot be absolute or escape via traversal.

`load_track_manifest(path)` requires:

- strict manifest JSON keys/schema;
- unique manifest `track_id` values;
- unique normalized relative paths;
- lowercase 64-character SHA-256 values;
- referenced file existence;
- loaded asset ID equals manifest ID;
- computed `world_sha256` equals manifest value;
- computed `content_sha256` equals manifest value.

Any mismatch fails. Loading never rewrites hashes or mutates the manifest.

Automatic manifest-generation/update tooling is not required by WP2.

## 17. Public API

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

Internal parser/canonicalization helpers remain private.

## 18. Scenario construction

WP2 does not redesign `F1RLEnvFactory`.

Later integration combines experiment-owned config with the asset:

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

## 19. WP2A — asset infrastructure

WP2A implements:

- immutable asset DTOs;
- strict loader;
- sandbox object construction;
- canonicalization and dual hashes;
- manifest model/loader/integrity checks;
- deterministic initial-state builder;
- documentation and focused tests.

WP2A tests use dedicated fixtures. It does not invent research tracks merely to test parsing.

## 20. WP2B — candidate track corpus

WP2B authors actual candidate project tracks to establish representative geometry scale:

```text
assets/tracks/
├── manifest.json
├── training/*.json
├── evaluation/*.json
└── demo/*.json
```

Current research direction is approximately four candidate training-side tracks and two candidate evaluation-side tracks. WP2 does not scientifically freeze that split.

Every checked-in research candidate must:

- load through production `load_track_asset`;
- match its manifest fingerprints;
- pass `RaceRoute.from_track`;
- provide enough real dimensional evidence for WP1B to choose environment/sensor/reward/episode numeric values.

## 21. Relationship to later work

### WP1B

Uses WP2B geometry scale to justify/freeze vehicle values, simulation timestep/surface distance, sensor layout/ranges, observation scales, reward weights, and episode horizon. Those values never enter track JSON.

### WP3

Owns scientific split/leakage validation. It can enforce disjoint track IDs and disjoint `world_sha256` values between training and held-out sets. Using a world hash that excludes `track_id` prevents rename-based leakage.

### WP6

Records `track_id`, `world_sha256`, and `content_sha256` as actual run provenance. WP2 computes stable identities; WP6 records what ran.

## 22. Error behavior

Invalid schema/content raises clear `ValueError` messages identifying the failing field/path. Missing/unreadable paths use normal filesystem errors unless the implementation plan finds a narrow reason for one project-owned exception.

WP2 does not create a broad error hierarchy.

## 23. Required tests

### Parser/schema

- valid complete asset loads;
- duplicate keys at any nesting level rejected;
- unknown/missing keys rejected;
- unsupported schema rejected;
- bool-as-number rejected;
- non-finite values rejected;
- wrong vector/quaternion lengths rejected.

### Domain construction

- all seven `TrackWorld` categories build correct sandbox types;
- existing world-wide ID uniqueness remains authoritative;
- invalid unit/orthogonality/extents/entity values fail.

### Canonicalization/hashing

- whitespace/key/entity input order does not affect hashes;
- source `1` vs `1.0` for continuous fields does not affect hashes;
- `q` vs `-q` does not affect hashes;
- negative zero does not affect hashes;
- track-ID-only change affects neither hash;
- spawn-only change affects only content hash;
- geometry/entity change affects both hashes;
- known canonical fixtures have pinned expected hashes.

### Initial state

- spawn preserved;
- velocity/angular velocity zero;
- score/mutable sets empty;
- grounded derives from surface query;
- no surface gives `grounded=False`;
- invalid max-surface-distance rejected.

### Manifest

- paths resolve relative to manifest;
- absolute/out-of-root paths rejected;
- duplicate IDs/paths rejected;
- malformed hashes rejected;
- ID mismatch and hash tampering detected.

### Integration/boundaries

- candidate research tracks are race-ready via `RaceRoute.from_track`;
- `sandbox` does not import `track_assets`;
- `track_assets` does not import `learning`, `runtime`, Gymnasium, SB3, Torch, API, or frontend code;
- SB3 production owner invariant remains exactly two files;
- production Torch imports remain zero;
- existing suite stays green.

## 24. Non-goals

WP2 does not implement:

- editor UI;
- arbitrary mesh/spline formats;
- binary/database/network asset storage;
- schema migration engine;
- save/export authoring API unless later needed by the editor;
- automatic manifest mutation;
- scientific split enforcement;
- sensor/reward/physics tuning;
- evaluation metrics;
- policy training/evaluation;
- runtime provenance collection.

## 25. Acceptance criteria

WP2 is complete when:

1. strict portable JSON assets construct validated sandbox domain objects;
2. only world/entities + spawn are authorable per track;
3. mutable initial state is canonical and deterministic;
4. `world_sha256` and `content_sha256` are stable, versioned semantic identities;
5. rename/reformat/reorder cannot disguise the same world geometry;
6. manifest integrity detects content drift/tampering;
7. real candidate track assets exist and pass production load + race-route validation;
8. sandbox/RL dependency boundaries remain intact;
9. real geometry scale is available for WP1B;
10. WP3 split semantics remain deliberately unfrozen.

## 26. Sequence after approval

```text
WP2 written spec
→ WP2 implementation plan
→ WP2A TDD implementation + verification
→ WP2B candidate track corpus + verification
→ WP2 full review
→ WP1B environment numeric freeze
→ WP3 train/held-out split contract
```

No WP1B or WP3 implementation begins inside WP2.
