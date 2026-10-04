# Run Provenance Design

**Status:** Written specification for review
**Branch:** `research/run-provenance`
**Base HEAD:** `8d1c2a7bd2ed22e01c580063f63806f2b4b17807`

## 1. Goal

WP6 adds a small, immutable provenance snapshot so every later scientific training/evaluation result can answer:

> **Exactly what code, experiment/config files, runtime stack, and track assets produced this result?**

WP6 is the reproducibility/audit layer between WP5 metrics and WP7 pilot training.

It does not train PPO, choose final training settings, evaluate held-out performance, change the protocol, or become an MLOps platform.

The core distinction is:

```text
WP1: what the experiment intends to run
WP2: stable identities for track content
WP3: train/held-out partition
WP4/WP5: evidence and measurement
WP6: what actually ran
```

## 2. Why WP6 must exist before WP7

The existing `TrainingRunner` already writes mutable `state.json` containing:
- run identity/status;
- requested/actual timesteps;
- resolved PPO/vector settings;
- sampling seed;
- training track IDs;
- checkpoint/final-model paths.

That is operational run state, not enough scientific provenance.

Without WP6, an old model cannot prove:
- which Git commit produced it;
- whether the worktree was dirty;
- which exact experiment/training YAML bytes were present;
- which Python/NumPy/Gymnasium/SB3/Torch/PyYAML versions were installed;
- which OS/platform executed it;
- whether a track with the same ID still had the same world/spawn hashes.

WP6 records those immutable facts before WP7 creates pilot artifacts.

## 3. Architecture

Keep provenance as one focused learning-layer module:

```text
f1rl/learning/provenance.py
```

Responsibilities:
- immutable provenance DTOs;
- Git state capture;
- exact source-file SHA-256 fingerprints;
- runtime/dependency version capture;
- selected track ID/hash resolution from the canonical manifest;
- deterministic conversion to a JSON document.

`TrainingRunner` remains responsible only for run artifact lifecycle. It may receive an already captured `RunProvenance` and persist it as:

```text
artifacts/runs/<run-id>/
├── state.json
├── provenance.json
├── checkpoints/
└── <final-model>.zip
```

Important boundary:

```text
provenance collector -> Git/files/config/manifest/runtime metadata
TrainingRunner       -> writes supplied immutable snapshot
```

The runner must not execute Git commands, inspect package metadata, load the experiment definition, or resolve track manifests itself.

## 4. Public provenance contract

Recommended immutable values:

```python
@dataclass(frozen=True, slots=True)
class FileFingerprint:
    role: str
    path: str
    sha256: str

@dataclass(frozen=True, slots=True)
class GitProvenance:
    commit_sha: str
    dirty: bool

@dataclass(frozen=True, slots=True)
class PackageVersion:
    name: str
    version: str

@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    python_version: str
    python_implementation: str
    system: str
    release: str
    machine: str
    packages: tuple[PackageVersion, ...]

@dataclass(frozen=True, slots=True)
class TrackProvenance:
    track_id: str
    world_sha256: str
    content_sha256: str

@dataclass(frozen=True, slots=True)
class RunProvenance:
    schema_version: int
    experiment_id: str
    protocol_version: int
    experiment_status: str
    training_protocol_status: str
    git: GitProvenance
    files: tuple[FileFingerprint, ...]
    runtime: RuntimeProvenance
    tracks: tuple[TrackProvenance, ...]
```

Exact implementation may use equivalent names while preserving these semantics.

All result objects contain only plain immutable Python values and are JSON-serializable without carrying `Path`, domain objects, environments, models, or subprocess objects.

## 5. Provenance schema version

WP6 provenance starts at:

```text
schema_version = 1
```

This is the schema of `provenance.json`, not the scientific protocol version.

A future incompatible provenance document shape requires a new provenance schema version.

The experiment's own:

```text
experiment_id
protocol_version
status
training.protocol_status
```

remain separate fields.

## 6. Experiment identity

WP6 records directly from the supplied `ExperimentDefinition`:

```text
experiment_id
protocol_version
experiment_status
training_protocol_status
```

WP6 does **not** require the experiment to already be fully frozen.

That is deliberate because the current canonical experiment is still:

```yaml
experiment:
  status: draft

training:
  protocol_status: pilot_owned
```

and WP7 pilot must run before the final training profile/seeds are frozen.

WP6 records the lifecycle state honestly instead of promoting it to `frozen`.

For WP8 final baseline/generalization, later readiness gates should require the appropriate frozen state.

## 7. Exact file fingerprints

WP6 fingerprints the exact bytes of three canonical source files:

1. experiment definition YAML;
2. referenced training runtime config;
3. referenced track manifest.

Each fingerprint is:

```text
SHA-256(path.read_bytes())
```

No YAML parse/reserialization occurs before hashing.

Consequences:
- whitespace/comment-only edits change the file hash;
- semantically equivalent but byte-different source files remain distinguishable;
- the fingerprint answers “which exact checked-in input file bytes were used?”;
- scientific semantic equivalence remains the responsibility of the typed config/track contracts, not this byte hash.

The files tuple order is fixed:

```text
experiment_definition
training_runtime_config
track_manifest
```

The stored `path` is repository-relative POSIX text, never a machine-specific absolute path.

Canonical provenance capture rejects required input files outside the repository root.

## 8. Why raw-byte hashes rather than semantic config hashes

Track assets already have semantic hashes because WP2 intentionally canonicalizes geometry/content.

Experiment/training YAML do not currently have a canonical semantic serialization contract.

WP6 must not invent one.

Therefore:
- YAML/config/manifest source files -> exact byte SHA-256;
- each selected track -> existing WP2 `world_sha256` and `content_sha256`.

This keeps ownership clean and avoids a second competing config schema.

## 9. Git provenance

WP6 records:

```text
git.commit_sha
git.dirty
```

`commit_sha` comes from:

```text
git rev-parse HEAD
```

Dirty state is derived from:

```text
git status --porcelain=v1 --untracked-files=all
```

Any non-empty porcelain output means:

```text
dirty = true
```

Tracked modifications, staged modifications, and untracked files therefore all mark the working tree dirty.

Ignored files do not.

Git capture fails explicitly when:
- the configured root is not a Git repository;
- `git` cannot execute;
- `HEAD` cannot be resolved;
- status cannot be inspected.

WP6 never silently substitutes `"unknown"`.

## 10. Dirty working-tree semantics

`git_dirty=True` is an **audit warning**, not proof of exact reproducibility.

WP6 does not capture:
- the dirty patch;
- untracked file contents;
- staged diff blobs;
- a synthetic dirty-tree hash.

Therefore an artifact with:

```json
"git": {
  "commit_sha": "...",
  "dirty": true
}
```

correctly means:

> the run came from this commit plus unrecorded local changes.

WP6 itself does not reject dirty runs because WP7 pilot/development may still need them recorded.

A later final-science readiness gate for WP8 should require `dirty == false`.

This avoids pretending a dirty run is reproducible while keeping pilot workflow practical.

## 11. Runtime and dependency provenance

Capture runtime metadata without importing SB3/Torch into the provenance module.

Python/platform fields use the standard library:

```text
python_version
python_implementation
system
release
machine
```

Do not record:
- username;
- hostname;
- home directory;
- repository absolute path;
- device serials;
- IP/network data.

Required package versions are resolved through `importlib.metadata.version()` for this fixed v1 ordered set:

```text
numpy
gymnasium
stable-baselines3
torch
PyYAML
```

The stored package names use the canonical distribution names above and preserve that order.

If a required distribution version cannot be resolved, provenance capture fails before a provenance-bound scientific run begins.

No CUDA/GPU capability claim is inferred from package presence.

## 12. Selected track provenance

The collector receives explicit:

```python
selected_track_ids: tuple[str, ...]
```

It loads the canonical manifest referenced by the experiment definition using the existing verified `load_track_manifest(...)`.

For each selected ID, in caller-provided order, it records:

```text
track_id
world_sha256
content_sha256
```

Rules:
- selected IDs must be a non-empty tuple;
- IDs must be non-blank and unique;
- every selected ID must exist in the manifest;
- hashes come from the verified manifest entry, never recomputed independently by WP6;
- tuple order is preserved.

WP6 does not infer whether a selected track is training or held-out.

The caller owns that decision through WP3 split semantics.

For WP7 pilot, callers pass exactly:

```text
definition.tracks.training_ids
```

For later evaluation, the same collector can snapshot an explicit held-out track selection without adding a second provenance system.

## 13. Collector API

Recommended public entry point:

```python
capture_run_provenance(
    *,
    repository_root: str | Path,
    experiment_path: str | Path,
    definition: ExperimentDefinition,
    selected_track_ids: tuple[str, ...],
) -> RunProvenance
```

The function:
1. validates arguments;
2. resolves and verifies repository root;
3. reloads `experiment_path` and requires equality with the supplied `definition`;
4. verifies the experiment/training-config/manifest paths are inside the repository;
5. hashes the exact three source files;
6. captures Git SHA + dirty state;
7. captures runtime/package versions;
8. loads and verifies the manifest;
9. resolves selected track hashes;
10. returns immutable `RunProvenance`.

Reload-and-compare prevents provenance from being captured from one YAML while the caller trains from a different in-memory `ExperimentDefinition`.

WP6 does not mutate any source file.

## 14. Determinism of the snapshot

For unchanged:
- Git HEAD/dirty state;
- exact source files;
- runtime/package versions;
- selected track IDs/hashes;
- platform;

two provenance captures produce equal `RunProvenance` values and byte-identical serialized JSON.

Do not include:
- capture timestamp;
- run ID;
- checkpoint path;
- current process ID;
- random UUID;
- absolute repository path.

Run lifecycle/time already belongs to `state.json` / run identity.

This makes provenance a stable fingerprint of execution inputs/environment rather than another mutable log.

## 15. JSON representation

The document is explicit and stable:

```json
{
  "schema_version": 1,
  "experiment": {
    "id": "ppo-generalization-v1",
    "protocol_version": 1,
    "status": "draft",
    "training_protocol_status": "pilot_owned"
  },
  "git": {
    "commit_sha": "...",
    "dirty": false
  },
  "files": [
    {
      "role": "experiment_definition",
      "path": "configs/experiments/ppo-generalization-v1.yml",
      "sha256": "..."
    },
    {
      "role": "training_runtime_config",
      "path": "config.yml",
      "sha256": "..."
    },
    {
      "role": "track_manifest",
      "path": "assets/tracks/manifest.json",
      "sha256": "..."
    }
  ],
  "runtime": {
    "python_version": "...",
    "python_implementation": "...",
    "system": "...",
    "release": "...",
    "machine": "...",
    "packages": [
      {"name": "numpy", "version": "..."}
    ]
  },
  "tracks": [
    {
      "track_id": "straight-sprint-v1",
      "world_sha256": "...",
      "content_sha256": "..."
    }
  ]
}
```

Because the serialized file records roles, `FileFingerprint` may include an explicit `role` field even if the internal implementation uses a dedicated structure.

The final schema must pin one representation and test exact keys.

## 16. JSON writer semantics

Provide a project-owned serializer/writer such as:

```python
write_run_provenance(path: str | Path, provenance: RunProvenance) -> None
```

Requirements:
- UTF-8;
- deterministic key/collection order;
- trailing newline;
- finite/simple JSON values only;
- parent directory must already exist;
- destination must not already exist;
- no silent overwrite;
- write failure propagates.

A provenance file is immutable after creation.

There is no `update_provenance(...)`.

## 17. TrainingRunner integration

Keep runner integration minimal and optional at the generic framework level.

Recommended constructor:

```python
TrainingRunner(
    config: TrainingRunConfig,
    *,
    provenance: RunProvenance | None = None,
)
```

Reason for optionality:
- generic/single-track framework tests and diagnostics predate authored track identity;
- `run_single(factory)` cannot prove a track hash because `F1RLEnvFactory` carries no authored track ID;
- WP7 canonical multi-track pilot will explicitly supply provenance.

When provenance is supplied:
1. run-directory collision checks happen normally;
2. `state.json` records:
   ```json
   "provenance_file": "provenance.json"
   ```
3. `provenance.json` is written before environment/model creation;
4. failure to write provenance prevents training from starting;
5. provenance is never updated after creation.

When provenance is absent:
```json
"provenance_file": null
```

Existing generic TrainingRunner behavior otherwise remains unchanged.

## 18. Multi-track consistency gate

For `TrainingRunner.run_multi(scenarios)`, when provenance is supplied, require:

```text
tuple(provenance track IDs)
==
tuple(scenario.track_id for scenario in scenarios.scenarios)
```

Order and IDs must match exactly.

This prevents a run from claiming hashes for tracks different from the scenarios actually supplied.

No hash recomputation belongs in the runner.

For `run_single(factory)`, the runner cannot prove authored identity with the current API; it only persists supplied provenance and does not invent a track ID.

WP7 uses the multi-track path, so canonical pilot integrity is covered.

## 19. TrainingRunResult integration

When useful for caller navigation, `TrainingRunResult` may gain:

```python
provenance_path: Path | None
```

This is an artifact pointer only.

It does not duplicate provenance content into the result.

Existing fields and model/checkpoint semantics remain unchanged.

## 20. Failure lifecycle

Provenance capture should occur **before** calling the runner.

Therefore Git/config/runtime/manifest failures happen before a run directory is created.

Inside `TrainingRunner`, when a supplied provenance snapshot cannot be written:
- no environment is created;
- no PPO model is created;
- no learning begins;
- run state is marked failed if `state.json` already exists;
- the original write exception is re-raised.

Once `provenance.json` exists, later training failure does not delete or mutate it.

A failed training run therefore retains the immutable provenance that explains the attempted execution.

## 21. Seeds and resolved training values

WP6 does not duplicate all resolved `TrainingRunConfig` fields into `provenance.json`.

They already live in `state.json`:
- resolved PPO configuration, including PPO seed;
- vector-environment configuration;
- sampling seed;
- requested/actual timesteps;
- training track IDs.

WP6 instead fingerprints the source training config file and binds it to code/runtime/track identities.

Together:

```text
state.json      = resolved operational run state
provenance.json = immutable execution identity/environment
```

This avoids two competing copies of the resolved training contract.

## 22. Relationship to WP7 training-profile freeze

The current experiment references mutable:

```text
config.yml
```

and declares:

```text
training.protocol_status = pilot_owned
```

WP7 pilot is allowed to use that configuration while WP6 records its exact byte hash.

After WP7 chooses justified runtime settings, the final scientific training profile is frozen into a stable experiment-owned file.

WP8 provenance will then fingerprint that final frozen profile instead.

WP6 therefore does not choose:
- final total timesteps;
- vector backend/count;
- checkpoint interval;
- PPO adjustment;
- final seed replicate set.

## 23. Relationship to held-out discipline

WP6 may record track IDs/hashes for an explicitly requested selection.

It does not:
- execute tracks;
- evaluate policies;
- calculate metrics;
- inspect held-out performance;
- choose parameters from held-out results.

Recording an asset identity is not outcome inspection.

WP7 pilot should select only training IDs.

WP8 later owns held-out execution.

## 24. Production dependency boundaries

WP6 may depend on:
- standard library;
- `ExperimentDefinition`;
- existing track manifest model/loader.

WP6 must not import:
- Stable-Baselines3;
- Torch as a Python module;
- Gym environments;
- PPOModel;
- vector environments;
- evaluation metrics;
- sandbox physics directly.

Package versions are read through `importlib.metadata`, not framework imports.

`sandbox`, `runtime`, and `track_assets` must not import provenance upward.

Training runner may import the project-owned `RunProvenance`/writer only.

## 25. Privacy / machine metadata boundary

Scientific provenance needs platform identity, not personal identity.

Explicitly exclude:
- username;
- hostname;
- exact device name;
- home path;
- absolute working directory;
- IP/MAC address;
- serial numbers;
- environment variable dumps.

Allowed platform metadata is limited to non-personal compatibility facts:

```text
Python version/implementation
OS system/release
machine architecture
package versions
```

## 26. Testing strategy

### Contract tests

Pin:
- frozen/slots DTOs;
- exact schema version 1;
- non-blank IDs/names;
- SHA-256 validation;
- Git SHA format;
- unique file roles/paths;
- unique track IDs;
- package order/name/version validation;
- plain immutable values only.

### File fingerprint tests

Use temporary files to prove:
- exact bytes determine SHA-256;
- whitespace change changes hash;
- same bytes at same repository-relative role reproduce the same fingerprint;
- outside-repository source path is rejected;
- absolute machine path is never serialized.

### Git tests

Use a dedicated temporary Git repository, not the project worktree.

Prove:
- clean repo captures current HEAD + `dirty=False`;
- tracked modification -> dirty;
- staged modification -> dirty;
- untracked file -> dirty;
- ignored file alone does not mark dirty;
- missing/not-a-repo Git capture fails.

No production mocking of Git responses.

### Runtime tests

Use injectable/private version lookup only if needed for deterministic unit tests.

Pin fixed package order and failure on missing required distribution.

Test doubles stay under tests only.

### Manifest/selection tests

Using real manifest contracts:
- selected training IDs resolve exact existing hashes;
- caller order preserved;
- duplicate/blank/missing IDs rejected;
- manifest tampering remains rejected by existing loader.

No held-out track execution occurs.

### Serialization tests

Pin:
- exact document keys;
- deterministic JSON bytes;
- UTF-8/trailing newline;
- no overwrite;
- no timestamp/run ID/absolute paths;
- round-trip JSON structure remains plain.

### TrainingRunner integration tests

Pin:
- no provenance -> existing behavior + `provenance_file=null`;
- supplied provenance -> immutable `provenance.json` created;
- state references it;
- multi-track IDs must match exactly;
- mismatch fails before environment/model construction;
- provenance write failure prevents learning;
- later training failure preserves provenance;
- result exposes path if adopted.

At least one tiny real TrainingRunner integration run should prove state + provenance + model artifacts coexist without changing policy behavior.

## 27. Expected production surface

Expected additions:

```text
f1rl/learning/provenance.py
```

Expected minimal modifications:

```text
f1rl/learning/training/runner.py
f1rl/learning/training/__init__.py   # only if public result/API changes require it
```

No changes expected in:
- sandbox;
- runtime;
- track asset hashing/loading semantics;
- Gym environment;
- observations/actions/rewards/race;
- PPO implementation;
- vector-environment implementation;
- WP5 evaluator/metrics;
- canonical track assets;
- experiment numeric values;
- train/held-out split.

If implementation requires those changes, stop and reassess scope.

## 28. Non-goals

WP6 does not implement:
- MLflow;
- Weights & Biases;
- TensorBoard;
- database experiment tracking;
- cloud artifact storage;
- model registry;
- remote worker metadata;
- hardware benchmarking;
- GPU telemetry;
- dirty patch archiving;
- dependency lockfile generation;
- environment recreation;
- Docker/Nix packaging;
- training;
- evaluation;
- statistical analysis;
- held-out outcome inspection;
- final training-profile selection;
- final seed selection.

## 29. Acceptance criteria

WP6 is complete when:
1. exact experiment/runtime-config/manifest bytes are SHA-256 fingerprinted;
2. repository-relative portable paths are recorded without personal absolute paths;
3. Git HEAD and dirty status are captured without silent fallback;
4. dirty includes tracked, staged, and untracked non-ignored changes;
5. required runtime/package versions are captured without SB3/Torch imports;
6. selected track IDs resolve verified `world_sha256` + `content_sha256`;
7. provenance capture is deterministic for unchanged inputs/environment;
8. `provenance.json` has pinned schema v1 and cannot be silently overwritten;
9. TrainingRunner can persist an immutable supplied snapshot beside `state.json`;
10. multi-track provenance IDs exactly match actual scenario IDs;
11. provenance failure prevents learning from starting;
12. later run failure preserves provenance;
13. existing state/checkpoint/model lifecycle remains intact;
14. no held-out execution occurs;
15. SB3/Torch dependency boundaries remain unchanged;
16. full repository verification remains green.

## 30. Sequence after approval

```text
WP6 written spec
-> WP6 implementation plan
-> TDD provenance value/file hashing
-> TDD Git/runtime capture
-> TDD manifest selection
-> TDD deterministic JSON writer
-> TDD TrainingRunner integration
-> whole-WP6 audit + exact verification
-> merge gate
-> WP7 Pilot Training
```

WP7 must not begin inside WP6.
