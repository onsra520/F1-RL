# Track Asset Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build strict, portable, reproducible JSON track assets with semantic fingerprints, manifest integrity checks, deterministic initial state derivation, and a six-track candidate corpus that gives WP1B real geometry scale without freezing WP3 split semantics.

**Architecture:** Add `f1rl.track_assets` as a small persistence adapter that depends only on the sandbox domain. Track files contain world/entities plus spawn pose; experiment/runtime state stays outside. Canonical semantic hashes identify world-only and world+spawn content independently of source formatting or `track_id`.

**Tech Stack:** Python 3.11+, stdlib `json`, `hashlib`, `pathlib`, existing frozen sandbox dataclasses, pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-10-01-track-asset-contract-design.md`

## Global Constraints

- Production dependency direction is `track_assets -> sandbox`; `track_assets` must not import `learning`, `runtime`, Gymnasium, SB3, Torch, FastAPI, or frontend/editor code.
- `sandbox` must not import `track_assets`.
- Track JSON schema v1 stores only world/entities plus spawn pose; no experiment config or mutable initial state.
- Source `schema_version` is integer `1`; booleans do not count as integers.
- All JSON object mappings are strict: required keys present, unknown keys rejected, duplicate keys rejected at every nesting level.
- Non-finite numbers and bool-as-number inputs are rejected.
- Entity input order is non-semantic; `TrackWorld` IDs remain globally unique across categories.
- Continuous semantic numbers canonicalize to finite floats; discrete `schema_version`, checkpoint `order`, and pickup `points` remain integers.
- Quaternion `q` and `-q`, `-0.0` and `0.0`, source `1` and `1.0`, JSON whitespace/key order, and entity-array input order must not change the relevant semantic fingerprint.
- `world_sha256` excludes `track_id` and spawn. `content_sha256` excludes `track_id` but includes spawn.
- Hash domains are `f1rl-track-world-v1` and `f1rl-track-content-v1`.
- Manifest paths are repository-portable POSIX relative paths using `/`; reject backslashes, absolute paths, empty paths, and paths resolving outside the manifest directory.
- Manifest loading never rewrites files or regenerates checked-in hashes.
- Candidate `training/` and `evaluation/` directory names are organizational only in WP2; scientific split semantics remain WP3-owned.
- Existing SB3 production owner count remains exactly two; production Torch imports remain zero.
- Mocks/fakes/test doubles remain in dedicated tests only, never production paths.
- Do not begin WP1B or WP3.

## Review Focus

1. Deeply nested duplicate JSON keys must fail before domain construction, not silently use the last value; Task 2 pins this.
2. Numeric ambiguity/overflow (`bool`, `NaN`, infinities, huge values, `1` vs `1.0`, `-0.0`) must be deterministic and cross-platform; Tasks 1 and 2 pin this.
3. Quaternion equivalence/sign canonicalization must be stable even when the first components are zero; Task 1 pins this with regression vectors.
4. Manifest path normalization must behave identically on Windows/Linux and reject `\`, traversal, and resolved escape; Task 4 pins this.
5. Rename/spawn/tamper cases must preserve or alter exactly the intended hash and be detected by the manifest; Tasks 1 and 4 pin this.

---

## File Structure

Create:

```text
f1rl/track_assets/
├── __init__.py          # public API only
├── model.py             # immutable SpawnPose/TrackAsset/manifest DTOs
├── canonical.py         # semantic payloads + SHA-256
├── loader.py            # strict track JSON parsing/domain construction
├── initial_state.py     # deterministic WorldState derivation
└── manifest.py          # strict manifest loading/integrity verification

tests/track_assets/
├── test_canonical.py
├── test_loader.py
├── test_initial_state.py
├── test_manifest.py
└── test_boundaries.py

tests/integration/test_track_asset_corpus.py

assets/tracks/manifest.json
assets/tracks/training/straight-sprint-v1.json
assets/tracks/training/slalom-v1.json
assets/tracks/training/right-turn-v1.json
assets/tracks/training/ramp-v1.json
assets/tracks/evaluation/left-turn-v1.json
assets/tracks/evaluation/offset-slalom-v1.json
```

Modify:

```text
assets/tracks/README.md
assets/tracks/training/README.md
assets/tracks/evaluation/README.md
f1rl/README.md
```

No new dependency is added to `pyproject.toml`.

---

### Task 1: Immutable Asset DTOs and Semantic Fingerprints

**Files:**
- Create: `f1rl/track_assets/model.py`
- Create: `f1rl/track_assets/canonical.py`
- Create: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_canonical.py`

**Interfaces:**
- Consumes: existing `Vec3`, `Quaternion`, `TrackWorld` and entity dataclasses.
- Produces:
  - `SpawnPose(position: Vec3, orientation: Quaternion)`
  - `TrackAsset(schema_version: int, track_id: str, spawn: SpawnPose, world: TrackWorld, world_sha256: str, content_sha256: str)`
  - internal `_build_track_asset(*, schema_version: int, track_id: str, spawn: SpawnPose, world: TrackWorld) -> TrackAsset`
  - internal canonical payload/hash helpers used by Task 2.

- [ ] **Step 1: Write failing DTO/hash tests**

Add tests that assert:
- `SpawnPose`/`TrackAsset` are frozen and validate exact owned types/non-empty ID/schema v1/hash format;
- manually equivalent worlds differing only in entity tuple input order produce identical hashes;
- changing only `track_id` changes neither hash;
- changing only spawn changes only `content_sha256`;
- changing a surface/entity semantic value changes both hashes;
- continuous semantic `1` and `1.0` canonicalize identically;
- `-0.0` and `0.0` canonicalize identically;
- normalized equivalent `q` and `-q` canonicalize identically, including a vector whose leading quaternion components are zero;
- canonical payloads contain exact hash-domain strings.

Before product code, compute at least one expected 64-char SHA-256 constant from an exact literal canonical JSON preimage using a one-off stdlib `hashlib.sha256(...)` command; paste that constant into the regression test. Do not derive that expected constant by calling product canonicalization.

- [ ] **Step 2: Run RED**

Run:

```bash
python -m pytest tests/track_assets/test_canonical.py -q
```

Expected: FAIL because `f1rl.track_assets` does not exist.

- [ ] **Step 3: Implement DTOs and canonicalization**

In `model.py`, implement the two frozen public DTOs with strict type/schema/hash validation.

In `canonical.py`, implement `_build_track_asset(...)` and deterministic semantic serialization using stdlib JSON with stable keys and no insignificant whitespace. Canonicalize each sandbox category explicitly rather than generic `dataclasses.asdict()` so future unrelated fields cannot silently enter the hash contract.

Quaternion sign rule: after normalization, choose the sign for which the first component whose absolute value is non-zero is positive; normalize every numeric `-0.0` to `0.0` before serialization.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python -m pytest tests/track_assets/test_canonical.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add f1rl/track_assets tests/track_assets/test_canonical.py
git commit -m "feat(track): define semantic track identities"
```

---

### Task 2: Strict Track JSON Loader and Sandbox Construction

**Files:**
- Create: `f1rl/track_assets/loader.py`
- Modify: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_loader.py`

**Interfaces:**
- Consumes: Task 1 `_build_track_asset(...)` and sandbox constructors.
- Produces: `load_track_asset(path: str | Path) -> TrackAsset`.

- [ ] **Step 1: Write failing loader tests**

Use temporary JSON files. Pin one complete asset containing all seven `TrackWorld` categories and assert exact sandbox types/values.

Add tests for:
- root and nested duplicate keys;
- root/nested unknown keys;
- missing required keys;
- schema values other than integer `1` and boolean `True`;
- empty/non-string `track_id`;
- wrong vector/quaternion lengths;
- bool-as-number in every numeric parser family;
- `NaN`, `Infinity`, `-Infinity` rejected by JSON parsing;
- huge continuous numeric input that cannot become a finite float produces `ValueError`, not platform-specific overflow leakage;
- invalid surface unit/orthogonality/dimensions propagate as clear field-context `ValueError`;
- invalid volume extents/entity numeric constraints fail;
- world-wide duplicate IDs across different entity categories fail through existing `TrackWorld` validation;
- whitespace, JSON mapping-key order, and entity-array order do not affect loaded hashes.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/track_assets/test_loader.py -q
```

Expected: FAIL because `load_track_asset` is missing.

- [ ] **Step 3: Implement strict JSON parsing**

Implement `load_track_asset(path: str | Path) -> TrackAsset` using stdlib `json`:
- UTF-8 text;
- `object_pairs_hook` that rejects duplicate keys at every object level;
- `parse_constant` rejection for non-standard non-finite constants;
- exact-key helpers for each schema object;
- continuous numbers converted to finite `float` after rejecting bool;
- discrete values remain strict `int` after rejecting bool.

Construct exactly:
- `PlanarSurfacePatch`;
- `Obstacle`;
- `BoostPad`;
- `SlowZone`;
- `Checkpoint`;
- `FinishLine`;
- `ScorePickup`;
- `OrientedBoxVolume`;
- `Vec3`/`Quaternion`;
- `TrackWorld`.

Then return `_build_track_asset(...)`. Do not add race-route validation here.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/track_assets/test_loader.py tests/track_assets/test_canonical.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add f1rl/track_assets tests/track_assets
git commit -m "feat(track): load strict portable track assets"
```

---

### Task 3: Deterministic Initial World State

**Files:**
- Create: `f1rl/track_assets/initial_state.py`
- Modify: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_initial_state.py`

**Interfaces:**
- Consumes: `TrackAsset`, sandbox `WorldState` and state dataclasses.
- Produces: `build_initial_world_state(asset: TrackAsset, *, max_surface_distance: float) -> WorldState`.

- [ ] **Step 1: Write failing initial-state tests**

Assert:
- exact spawn position/orientation preserved;
- velocity/angular velocity are zero vectors;
- score is zero;
- collected pickup IDs, active checkpoint IDs, active finish IDs are empty immutable sets;
- spawn exactly on a surface yields `grounded=True` for distance `0.0`;
- spawn outside any valid surface query yields `grounded=False`;
- builder never snaps/mutates spawn position;
- bool, negative, NaN, and infinite `max_surface_distance` values fail.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/track_assets/test_initial_state.py -q
```

Expected: FAIL because builder is missing.

- [ ] **Step 3: Implement builder**

Construct `WorldState` with canonical empty `TrackState`, `InteractionState`, and `ScoreState(0)`. Derive only `grounded` via `asset.world.surface_at(asset.spawn.position, max_surface_distance) is not None`.

Do not import `SimulationConfig`, `learning`, or runtime.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/track_assets/test_initial_state.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add f1rl/track_assets tests/track_assets/test_initial_state.py
git commit -m "feat(track): derive canonical track initial state"
```

---

### Task 4: Manifest Registry and Integrity Verification

**Files:**
- Modify: `f1rl/track_assets/model.py`
- Create: `f1rl/track_assets/manifest.py`
- Modify: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_manifest.py`

**Interfaces:**
- Consumes: `load_track_asset(path)` from Task 2.
- Produces:
  - `TrackManifestEntry(track_id: str, path: str, world_sha256: str, content_sha256: str)`
  - `TrackManifest(schema_version: int, tracks: tuple[TrackManifestEntry, ...])`
  - `load_track_manifest(path: str | Path) -> TrackManifest`.

- [ ] **Step 1: Write failing manifest tests**

Pin:
- strict root/entry keys and schema v1;
- nested duplicate-key rejection;
- unique track IDs;
- unique normalized relative paths;
- lowercase exactly-64 hex hashes;
- manifest paths must use `/`; reject any `\\` path even on Windows;
- empty/absolute/drive-qualified paths rejected;
- `.`/`..` normalization that escapes manifest root rejected;
- existing symlink or resolved path escape is rejected where the platform supports symlinks; skip only when the test environment cannot create one;
- resolution is relative to manifest location, not process CWD;
- referenced target must be a file;
- loaded asset ID mismatch fails;
- world/content hash mismatch fails independently;
- valid manifest loads without modifying its bytes or track files.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/track_assets/test_manifest.py -q
```

Expected: FAIL because manifest API is missing.

- [ ] **Step 3: Implement manifest DTOs and loader**

Use the same strict duplicate-key JSON decoder behavior as Task 2; refactor the decoder into a private helper only if needed to avoid duplication without creating a generic framework.

Normalize POSIX manifest paths with `PurePosixPath`, reject backslashes/absolute/drive-like prefixes/traversal, then resolve under the manifest directory and verify the resolved file remains inside the manifest root before loading it.

Return immutable manifest DTOs only after every referenced asset passes ID and both hash checks. Do not auto-update hashes.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/track_assets/test_manifest.py tests/track_assets -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add f1rl/track_assets tests/track_assets
git commit -m "feat(track): verify track asset manifests"
```

---

### Task 5: Author the Candidate Six-Track Corpus

**Files:**
- Create: `assets/tracks/manifest.json`
- Create: `assets/tracks/training/straight-sprint-v1.json`
- Create: `assets/tracks/training/slalom-v1.json`
- Create: `assets/tracks/training/right-turn-v1.json`
- Create: `assets/tracks/training/ramp-v1.json`
- Create: `assets/tracks/evaluation/left-turn-v1.json`
- Create: `assets/tracks/evaluation/offset-slalom-v1.json`
- Modify: `assets/tracks/README.md`
- Modify: `assets/tracks/training/README.md`
- Modify: `assets/tracks/evaluation/README.md`
- Test: `tests/integration/test_track_asset_corpus.py`

**Interfaces:**
- Consumes: `load_track_manifest`, `load_track_asset`, existing `RaceRoute.from_track`.
- Produces: six candidate data assets and one checked-in integrity manifest. These are dimensional evidence for WP1B, not a frozen WP3 research split.

Candidate geometry contract:

| Asset | Core world geometry | Route/entity intent |
|---|---|---|
| `straight-sprint-v1` | one horizontal surface, width 12 m, length 120 m, centered on +Z | spawn near Z=4; checkpoints near Z=35 and 75; finish near Z=112; no obstacle forcing |
| `slalom-v1` | one horizontal surface, width 14 m, length 130 m | three laterally alternating checkpoint targets and four alternating obstacle boxes; finish near far end |
| `right-turn-v1` | two overlapping horizontal 12×70 m surfaces: +Z leg then +X leg | checkpoints before and after 90° right turn; finish near +X end |
| `ramp-v1` | 12 m wide: 50 m horizontal approach, 60 m +8° incline, 50 m upper horizontal | at least one checkpoint on incline and one on upper deck; finish on upper deck |
| `left-turn-v1` | mirror of right-turn topology using a distinct -X second leg | unseen turn direction; checkpoint/finish volumes mirrored, not copied with same geometry hash |
| `offset-slalom-v1` | one horizontal surface, width 16 m, length 140 m | unseen lateral target/obstacle spacing distinct from training slalom |

For `ramp-v1`, pin the incline basis from `theta = 8°`:

```text
forward = [0, sin(theta), cos(theta)]
normal  = [0, cos(theta), -sin(theta)]
```

The lower endpoint meets the approach at `(0, 0, 50)`. The 60 m incline therefore ends at approximately `(0, 8.3503860576, 109.4160841245)`. The upper horizontal starts there and continues +Z for 50 m. Use these same derived values in the checked-in JSON; do not introduce a runtime generator.

All candidate spawns use zero-height surface contact (`y` exactly on authored surface) and identity orientation unless the route begins along another axis. Checkpoint/finish volumes are 4 m tall and span the usable road width while remaining thin (2 m full thickness) along travel direction. Obstacles remain at least 2 m wide and leave a traversable corridor at least 4 m wide.

- [ ] **Step 1: Write failing corpus integration tests**

Assert exact expected six manifest IDs/paths, then for every entry:
- production manifest load succeeds;
- `load_track_asset` succeeds;
- manifest hashes equal computed hashes;
- `RaceRoute.from_track(asset.world)` succeeds;
- each asset has at least one surface and exactly one finish line;
- dimensions are finite/positive and overall extents demonstrate: road widths 12–16 m, longitudinal scale at least 120 m, and vertical elevation >8 m in `ramp-v1`;
- all six `world_sha256` values are unique;
- all six `content_sha256` values are unique.

Also assert the directory labels are not represented as scientific split fields in manifest JSON.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tests/integration/test_track_asset_corpus.py -q
```

Expected: FAIL because manifest/corpus files do not exist.

- [ ] **Step 3: Author the six JSON assets**

Use only schema-v1 world/spawn fields. Keep entity IDs descriptive and unique inside each world. Do not add sensor/reward/physics/episode/training fields.

The candidate tracks are intentionally simple primitives; do not add spline/mesh/editor formats.

- [ ] **Step 4: Compute and write manifest hashes explicitly**

After assets load, use a one-off local command that imports `load_track_asset` to print each file's two computed hashes. Paste those values into `assets/tracks/manifest.json` manually. Do not commit an auto-generation script.

- [ ] **Step 5: Run GREEN and inspect dimensions**

```bash
python -m pytest tests/integration/test_track_asset_corpus.py tests/track_assets -q
```

Expected: PASS.

Update track README files to state that `training/`/`evaluation/` are candidate organizational buckets until WP3 and record the corpus scale summary (width 12–16 m, route length roughly 120–160 m, ramp rise ~8.35 m).

- [ ] **Step 6: Commit**

```bash
git add assets/tracks tests/integration/test_track_asset_corpus.py
git commit -m "feat(track): add candidate research track corpus"
```

---

### Task 6: Dependency Boundaries, Documentation, and Full WP2 Verification

**Files:**
- Create: `tests/track_assets/test_boundaries.py`
- Modify: `f1rl/README.md`
- Modify: `f1rl/track_assets/__init__.py` only if public exports need correction

**Interfaces:**
- Consumes: complete WP2 implementation.
- Produces: final architectural regression gate and documented package boundary.

- [ ] **Step 1: Write/complete boundary tests**

Use AST/import inspection consistent with existing architecture tests and assert:
- no file under `f1rl/sandbox` imports `f1rl.track_assets`;
- no production `f1rl/track_assets/*.py` imports `f1rl.learning`, `f1rl.runtime`, Gymnasium, SB3, Torch, FastAPI, or frontend code;
- public package exports exactly the intended seven names: `SpawnPose`, `TrackAsset`, `TrackManifest`, `TrackManifestEntry`, `load_track_asset`, `load_track_manifest`, `build_initial_world_state`;
- SB3 production import owners remain exactly `f1rl/learning/algorithms/ppo.py` and `f1rl/learning/training/vector_env.py`;
- direct production Torch imports remain zero;
- production `Mock`/`Fake` markers remain zero.

- [ ] **Step 2: Run focused WP2 suite**

```bash
python -m pytest tests/track_assets tests/integration/test_track_asset_corpus.py -q
```

Expected: PASS.

- [ ] **Step 3: Run learning/orchestration regression suite**

```bash
python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full suite and static gates**

```bash
python -m pytest -q
python -m compileall -q f1rl scripts
python -m pip check
git diff --check 64334cabfd50dd706b286f86d06c15330368ec30...HEAD
git status --short
```

Expected:
- all tests pass;
- compileall clean;
- no broken requirements;
- no whitespace errors;
- worktree clean after commits.

- [ ] **Step 5: Whole-WP2 review**

Review the entire diff from spec HEAD `64334cabfd50dd706b286f86d06c15330368ec30` to candidate HEAD for Critical/Important defects. Specifically inspect:
- whether any track-owned file contains experiment config or mutable runtime state;
- hash preimage stability and domain separation;
- path traversal/cross-platform behavior;
- candidate world-hash uniqueness;
- whether any code accidentally begins WP1B/WP3.

If a Critical/Important issue is found: add a dedicated regression test, fix it TDD-style, rerun focused/full gates, and commit the fix separately.

- [ ] **Step 6: Commit final documentation/boundary changes**

```bash
git add f1rl/README.md tests/track_assets/test_boundaries.py f1rl/track_assets/__init__.py
git commit -m "docs(track): document portable track asset boundary"
```

Then rerun the exact final verification commands on the final HEAD before claiming WP2 complete.

---

## Stop Condition

When Task 6 is fully verified:

- report exact commits and final HEAD;
- report focused/full test counts and static/boundary results;
- summarize the six candidate track scales and hashes;
- report any remaining non-blocking issues;
- **STOP**.

Do not begin WP1B numeric freeze or WP3 split implementation without a new reviewed gate.