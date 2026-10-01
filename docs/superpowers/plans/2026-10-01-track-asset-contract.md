# Track Asset Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build strict, portable, reproducible JSON track assets with semantic fingerprints, manifest integrity checks, deterministic initial-state derivation, and a six-track candidate corpus that gives WP1B real geometry scale without freezing WP3 split semantics.

**Architecture:** Add `f1rl.track_assets` as a small persistence adapter that depends only on the sandbox domain. Track files own world/entities plus spawn pose; experiment/runtime state stays outside. Canonical semantic hashes identify world-only and world+spawn content independently of source formatting or `track_id`.

**Tech Stack:** Python 3.11+, stdlib `json`, `hashlib`, `pathlib`, existing frozen sandbox dataclasses, pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-10-01-track-asset-contract-design.md`

## Global Constraints

- Production direction is `track_assets -> sandbox`; `track_assets` must not import `learning`, `runtime`, Gymnasium, SB3, Torch, FastAPI, or frontend/editor code.
- `sandbox` must not import `track_assets`.
- Track schema v1 stores only world/entities + spawn pose. No experiment config or mutable initial state.
- Source `schema_version` is strict integer `1`; booleans do not count as integers.
- Every JSON mapping is strict: missing, unknown, and duplicate keys fail at any nesting level.
- Reject non-finite numbers and bool-as-number input.
- Continuous semantics canonicalize to finite floats; `schema_version`, checkpoint `order`, and pickup `points` remain integers.
- Entity array input order is non-semantic; `TrackWorld` keeps global ID uniqueness authoritative.
- `q/-q`, `-0.0/0.0`, source `1/1.0`, whitespace/key order, and entity-array order must not alter semantic identity.
- `world_sha256` excludes `track_id` and spawn; `content_sha256` excludes `track_id` but includes spawn.
- Hash domains are exactly `f1rl-track-world-v1` and `f1rl-track-content-v1`.
- Manifest paths use portable POSIX `/`; reject `\`, absolute/drive-qualified/empty paths, traversal, and resolved escape.
- Manifest loading never mutates files or rewrites hashes.
- `training/` and `evaluation/` remain organizational labels only until WP3.
- SB3 production owners remain exactly two; direct production Torch imports remain zero.
- Test doubles stay in dedicated tests only.
- Do not begin WP1B or WP3.

## Review Focus

1. Deep nested duplicate JSON keys must fail before domain construction — Task 2.
2. Numeric ambiguity/overflow (`bool`, NaN/Inf, huge values, `1/1.0`, `-0.0`) must be deterministic cross-platform — Tasks 1–2.
3. Quaternion sign canonicalization must remain stable when leading components are zero — Task 1.
4. Manifest paths must behave identically on Windows/Linux and reject backslash/traversal/symlink escape — Task 4.
5. Rename/spawn/tamper cases must change or preserve exactly the intended hash and be caught by manifest verification — Tasks 1 and 4.

---

## File Structure

Create:

```text
f1rl/track_assets/
├── __init__.py
├── model.py
├── canonical.py
├── loader.py
├── initial_state.py
└── manifest.py

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

Modify track READMEs and `f1rl/README.md`. Do not add a dependency to `pyproject.toml`.

---

### Task 1: Immutable Asset DTOs and Semantic Fingerprints

**Files:**
- Create: `f1rl/track_assets/model.py`
- Create: `f1rl/track_assets/canonical.py`
- Create: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_canonical.py`

**Interfaces:**
- Consumes existing `Vec3`, `Quaternion`, `TrackWorld`, and entity dataclasses.
- Produces:
  - `SpawnPose(position: Vec3, orientation: Quaternion)`
  - `TrackAsset(schema_version: int, track_id: str, spawn: SpawnPose, world: TrackWorld, world_sha256: str, content_sha256: str)`
  - internal `_build_track_asset(*, schema_version: int, track_id: str, spawn: SpawnPose, world: TrackWorld) -> TrackAsset`.

- [ ] **Step 1: Write RED tests**

Pin frozen/type/schema/hash validation plus these hash invariants: entity tuple reorder unchanged; `track_id`-only change affects neither hash; spawn-only change affects only content hash; geometry/entity change affects both; `1/1.0`, `-0.0/0.0`, and `q/-q` are equivalent. Include a quaternion whose first non-zero component is not `w`.

Before product code, compute at least one expected SHA-256 constant from an exact literal canonical JSON preimage with one-off stdlib `hashlib`; paste the constant into the test. Do not obtain it from product canonicalization.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/track_assets/test_canonical.py -q
```

Expected: import/API failure because `f1rl.track_assets` does not exist.

- [ ] **Step 3: Implement minimal DTO/hash code**

Canonicalize each sandbox category explicitly, not through generic `asdict()`. JSON output uses UTF-8 semantics, sorted mapping keys, compact separators, normalized finite floats, stable integer fields, and exact hash-domain fields. Quaternion sign: after normalization, the first non-zero component in `(w,x,y,z)` is positive.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/track_assets/test_canonical.py -q
git add f1rl/track_assets tests/track_assets/test_canonical.py
git commit -m "feat(track): define semantic track identities"
```

---

### Task 2: Strict Track JSON Loader and Sandbox Construction

**Files:**
- Create: `f1rl/track_assets/loader.py`
- Modify: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_loader.py`

**Interface:** `load_track_asset(path: str | Path) -> TrackAsset`.

- [ ] **Step 1: Write RED tests**

Use temp JSON. One valid fixture must exercise all seven `TrackWorld` categories and assert exact sandbox types/values. Add failures for root/nested duplicate keys, missing/unknown keys, unsupported/bool schema, bad ID, wrong vector/quaternion length, bool-as-number, `NaN/Infinity/-Infinity`, huge values that cannot become finite floats, invalid surface basis/dimensions, invalid extents/entity values, and cross-category duplicate IDs. Assert missing path remains `FileNotFoundError` and invalid UTF-8 remains `UnicodeDecodeError` rather than being wrapped in a broad custom hierarchy. Assert formatting/key/entity order does not alter hashes.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/track_assets/test_loader.py -q
```

- [ ] **Step 3: Implement loader**

Use stdlib `json` with duplicate-key rejecting `object_pairs_hook` and non-finite rejecting `parse_constant`. Use exact-key helpers. Convert continuous numbers to finite float after bool rejection; retain discrete integers. Construct existing `PlanarSurfacePatch`, `OrientedBoxVolume`, all six volume-backed entity types, `Vec3`, `Quaternion`, and `TrackWorld`; return Task 1 `_build_track_asset(...)`. Do not validate `RaceRoute` here.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/track_assets/test_loader.py tests/track_assets/test_canonical.py -q
git add f1rl/track_assets tests/track_assets
git commit -m "feat(track): load strict portable track assets"
```

---

### Task 3: Deterministic Initial World State

**Files:**
- Create: `f1rl/track_assets/initial_state.py`
- Modify: `f1rl/track_assets/__init__.py`
- Test: `tests/track_assets/test_initial_state.py`

**Interface:** `build_initial_world_state(asset: TrackAsset, *, max_surface_distance: float) -> WorldState`.

- [ ] **Step 1: Write RED tests**

Assert exact spawn pose preservation; zero linear/angular velocity; score zero; empty collected/active immutable sets; on-surface spawn gives `grounded=True` at distance `0.0`; no surface gives false; no spawn snapping; bool/negative/NaN/Inf distance rejected.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/track_assets/test_initial_state.py -q
```

- [ ] **Step 3: Implement minimal builder**

Construct `WorldState` with `TrackState(frozenset())`, `InteractionState(frozenset(), frozenset())`, and `ScoreState(0)`. Derive only grounded via `asset.world.surface_at(asset.spawn.position, max_surface_distance) is not None`. Do not import runtime or learning.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/track_assets/test_initial_state.py -q
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
- `TrackManifestEntry(track_id: str, path: str, world_sha256: str, content_sha256: str)`
- `TrackManifest(schema_version: int, tracks: tuple[TrackManifestEntry, ...])`
- `load_track_manifest(path: str | Path) -> TrackManifest`.

- [ ] **Step 1: Write RED tests**

Pin strict root/entry schema, duplicate-key rejection, unique IDs and normalized paths, lowercase exact-64 hex hashes, POSIX `/` only, rejection of backslashes/absolute/drive-qualified/empty/traversal paths, symlink/resolved escape where supported, CWD-independent resolution, target-is-file, ID mismatch, independent world/content hash mismatch, and no byte mutation on successful load.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/track_assets/test_manifest.py -q
```

- [ ] **Step 3: Implement manifest model/loader**

Reuse the strict JSON decoder privately if useful. Parse manifest paths with `PurePosixPath`; reject portable-path violations before mapping parts under `manifest_path.parent`. Resolve and assert the target remains under the resolved manifest root before loading. Return immutable DTOs only after every referenced asset matches ID and both hashes. Never auto-update.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/track_assets/test_manifest.py tests/track_assets -q
git add f1rl/track_assets tests/track_assets
git commit -m "feat(track): verify track asset manifests"
```

---

### Task 5: Author the Candidate Six-Track Corpus

**Files:** the six JSON paths and `assets/tracks/manifest.json` listed above; modify track READMEs; test `tests/integration/test_track_asset_corpus.py`.

**Interfaces:** consumes production loaders plus existing `RaceRoute.from_track`. Every `track_id` is exactly the filename stem.

#### Exact candidate geometry

All unmentioned entity categories are empty. All volume orientations are identity quaternion `[1,0,0,0]`. All spawns are identity orientation and lie exactly on the authored surface (`y` equals surface height). Finish/checkpoint `order` starts at zero and is contiguous.

**`straight-sprint-v1`**
- surface `road-main`: origin `[0,0,60]`, forward `[0,0,1]`, normal `[0,1,0]`, width `12`, length `120`;
- spawn `[0,0,4]`;
- checkpoints: centers `[0,0,35]`, `[0,0,75]`, each half-extents `[6,2,1]`;
- finish center `[0,0,112]`, half-extents `[6,2,1]`.

**`slalom-v1`**
- surface: origin `[0,0,65]`, forward +Z, normal +Y, width `14`, length `130`;
- spawn `[0,0,4]`;
- checkpoint centers `[-3,0,30]`, `[3,0,70]`, `[-3,0,110]`, each half-extents `[2,2,1.5]`;
- obstacle centers `[3.5,1,45]`, `[-3.5,1,60]`, `[3.5,1,85]`, `[-3.5,1,100]`, each half-extents `[2,1,2]`;
- finish `[0,0,124]`, half-extents `[7,2,1]`.

**`right-turn-v1`**
- `road-entry`: origin `[0,0,35]`, forward +Z, normal +Y, width `12`, length `70`;
- `road-exit`: origin `[35,0,70]`, forward +X, normal +Y, width `12`, length `70`;
- spawn `[0,0,4]`;
- checkpoint 0 center `[0,0,35]`, half-extents `[6,2,1]`;
- checkpoint 1 center `[20,0,70]`, half-extents `[1,2,6]`;
- finish ` [64,0,70]`, half-extents `[1,2,6]`.

**`ramp-v1`**
- approach: origin `[0,0,25]`, forward +Z, normal +Y, width `12`, length `50`;
- incline lower endpoint is `[0,0,50]`, length `60`, width `12`, angle `8°`;
- incline forward `[0,0.13917310096006544,0.9902680687415704]`;
- incline normal `[0,0.9902680687415704,-0.13917310096006544]`;
- incline origin `[0,4.175193028801963,79.7080420622471]`;
- incline upper endpoint `[0,8.350386057603925,109.41608412449422]`;
- upper road: origin `[0,8.350386057603925,134.41608412449422]`, forward +Z, normal +Y, width `12`, length `50`;
- spawn `[0,0,4]`;
- checkpoint 0 lies 25 m along incline: center `[0,3.479327524001636,74.75670171853926]`, half-extents `[6,2,1]`;
- checkpoint 1 center `[0,8.350386057603925,130]`, half-extents `[6,2,1]`;
- finish center `[0,8.350386057603925,153]`, half-extents `[6,2,1]`.

**`left-turn-v1`**
- entry surface identical dimensions/position to `right-turn-v1` entry;
- exit surface origin `[-35,0,70]`, forward `[-1,0,0]`, normal +Y, width `12`, length `70`;
- spawn `[0,0,4]`;
- checkpoint 0 `[0,0,35]`, half-extents `[6,2,1]`;
- checkpoint 1 `[-20,0,70]`, half-extents `[1,2,6]`;
- finish `[-64,0,70]`, half-extents `[1,2,6]`.

**`offset-slalom-v1`**
- surface: origin `[0,0,70]`, forward +Z, normal +Y, width `16`, length `140`;
- spawn `[0,0,4]`;
- checkpoint centers `[4,0,28]`, `[-4,0,62]`, `[2.5,0,98]`, `[-3.5,0,124]`, each half-extents `[2,2,1.5]`;
- obstacle centers `[-4.5,1,45]`, `[4.5,1,72]`, `[-4,1,100]`, `[4,1,116]`, each half-extents `[2,1,2]`;
- finish `[0,0,134]`, half-extents `[8,2,1]`.

These are candidate primitive tracks, not claims of final scientific difficulty. They establish width `12–16 m`, longitudinal scale about `120–160 m`, 90° turns, lateral obstacle/checkpoint variation, and ~`8.35 m` elevation for WP1B.

- [ ] **Step 1: Write RED corpus integration tests**

Assert exact six IDs/paths, successful production manifest/asset load, matching hashes, `RaceRoute.from_track` success, exactly one finish per candidate, expected surface counts, scale envelope (`12–16 m` widths, >=120 m route extent, ramp rise >8 m), and unique world/content hashes. Assert manifest JSON contains no scientific split field.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/integration/test_track_asset_corpus.py -q
```

- [ ] **Step 3: Author the six JSON assets exactly as pinned above**

Use descriptive unique IDs. Do not add sensor/reward/physics/episode/training fields or runtime mutable state.

- [ ] **Step 4: Fill manifest hashes explicitly**

Run a one-off command importing `load_track_asset`, print both hashes for each candidate, and paste them into `manifest.json`. Do not commit a generator.

- [ ] **Step 5: Verify GREEN, update docs, commit**

```bash
python -m pytest tests/integration/test_track_asset_corpus.py tests/track_assets -q
git add assets/tracks tests/integration/test_track_asset_corpus.py
git commit -m "feat(track): add candidate research track corpus"
```

Track READMEs must state that directory names are candidate organizational buckets until WP3 and summarize the scale envelope above.

---

### Task 6: Dependency Boundaries and Full WP2 Verification

**Files:**
- Create: `tests/track_assets/test_boundaries.py`
- Modify: `f1rl/README.md`
- Modify `f1rl/track_assets/__init__.py` only if exports need correction.

- [ ] **Step 1: Pin boundary tests**

Use AST/import inspection consistent with existing architecture tests. Assert no sandbox→track_assets import; no track_assets→learning/runtime/Gymnasium/SB3/Torch/FastAPI/frontend import; public exports exactly `SpawnPose`, `TrackAsset`, `TrackManifest`, `TrackManifestEntry`, `load_track_asset`, `load_track_manifest`, `build_initial_world_state`; SB3 owners exactly the existing two files; production Torch imports zero; production Mock/Fake markers zero.

- [ ] **Step 2: Focused verification**

```bash
python -m pytest tests/track_assets tests/integration/test_track_asset_corpus.py -q
```

- [ ] **Step 3: Learning/orchestration regression**

```bash
python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py -q
```

- [ ] **Step 4: Full/static verification**

```bash
python -m pytest -q
python -m compileall -q f1rl scripts
python -m pip check
git diff --check 64334cabfd50dd706b286f86d06c15330368ec30...HEAD
git status --short
```

Expected: all tests pass, compileall clean, no broken requirements, no whitespace error, clean worktree after commits.

- [ ] **Step 5: Whole-WP2 review**

Review spec HEAD `64334cabfd50dd706b286f86d06c15330368ec30...HEAD` for Critical/Important defects, focusing on accidental experiment/runtime state in assets, hash stability/domain separation, path portability/traversal, candidate hash uniqueness, and accidental WP1B/WP3 scope creep. Any Important/Critical issue gets a dedicated regression test and separate fix commit, then all gates rerun.

- [ ] **Step 6: Final docs/boundary commit and exact-final rerun**

```bash
git add f1rl/README.md tests/track_assets/test_boundaries.py f1rl/track_assets/__init__.py
git commit -m "docs(track): document portable track asset boundary"
```

Rerun Steps 2–4 on the exact final HEAD before claiming completion.

---

## Stop Condition

After Task 6: report exact commits/final HEAD, focused/full test counts, compile/pip/diff/boundary results, six candidate scales + hashes, and remaining non-blocking issues; then **STOP**. Do not begin WP1B or WP3 without the next reviewed gate.