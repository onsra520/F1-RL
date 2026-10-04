# Run Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP6 as a deterministic, immutable run-provenance snapshot capturing exact experiment/config/manifest bytes, Git state, runtime/package versions, and selected track identities; integrate only persistence/consistency into `TrainingRunner`, without starting WP7 training.

**Architecture:** Add one focused `f1rl.learning.provenance` module. It owns immutable provenance DTOs, repository-relative exact-byte file fingerprints, Git/runtime/package capture, verified manifest track selection, deterministic JSON conversion/writing, and the public `capture_run_provenance(...)` entry point. `TrainingRunner` may receive an already captured snapshot, validate multi-track ID consistency, write immutable `provenance.json`, and reference it from `state.json`; it must not capture Git/package/manifest facts itself.

**Tech Stack:** Python 3.12, standard library (`dataclasses`, `hashlib`, `importlib.metadata`, `json`, `platform`, `subprocess`), existing `ExperimentDefinition` and track-manifest APIs, pytest. No direct Stable-Baselines3 or Torch imports in provenance.

**Spec:** `docs/superpowers/specs/2026-10-04-run-provenance-design.md`

## Global Constraints

- Branch: `research/run-provenance`.
- Base HEAD: `8d1c2a7bd2ed22e01c580063f63806f2b4b17807`.
- Approved spec commit: `91956113e255efc7e5b9bbec6d4b9132fae87a86`.
- Expected new production file:
  - `f1rl/learning/provenance.py`
- Expected minimal production modifications:
  - `f1rl/learning/training/runner.py`
  - `f1rl/learning/training/__init__.py` only if the changed result/API needs export updates.
- Do not modify:
  - `f1rl/sandbox/**`;
  - `f1rl/runtime/**`;
  - `f1rl/track_assets/**` hashing/loading semantics;
  - `f1rl/learning/gym_environment.py`;
  - observation/action/reward/race code;
  - PPO implementation;
  - vector-environment implementation;
  - WP5 evaluator/metrics;
  - canonical track assets;
  - canonical experiment numeric values or split membership.
- Do not train PPO for research, inspect held-out outcomes, select final training settings/seeds, or begin WP7.
- No production mocks/fakes/test switches.
- Any reusable test doubles stay under dedicated `tests/**/mocks/` modules.
- Direct production SB3 import owners remain exactly:
  - `f1rl/learning/algorithms/ppo.py`
  - `f1rl/learning/training/vector_env.py`
- Direct production Torch imports remain zero.
- Provenance capture must not import SB3/Torch modules to determine versions; use `importlib.metadata`.

## Execution Environment

The WP6 worktree currently has no local `.venv`. Use the previously verified interpreter:

```powershell
..\wp4-evaluation-telemetry\.venv\Scripts\python.exe
```

Resolve it without falling back to system/Conda Python:

```powershell
$python = if (Test-Path '.\.venv\Scripts\python.exe') {
    (Resolve-Path '.\.venv\Scripts\python.exe').Path
} else {
    (Resolve-Path '..\wp4-evaluation-telemetry\.venv\Scripts\python.exe').Path
}
```

Then:

```powershell
& $python -m pytest ...
```

All Git subprocess tests must create their own temporary Git repositories. They must never mutate or dirty the project worktree.

## Review Focus

1. Exact-byte config fingerprints, not invented semantic YAML hashes.
2. Repository-relative POSIX paths only; no personal absolute paths.
3. Git dirty state includes staged/tracked/untracked non-ignored changes.
4. Dirty state is recorded, not treated as exact reproducibility.
5. Package versions are read without importing SB3/Torch.
6. Track hashes come from the verified existing manifest contract.
7. `capture_run_provenance` reloads the experiment YAML and compares it with the supplied definition.
8. Provenance contains no timestamp/run ID/process ID/random UUID.
9. `provenance.json` is immutable/no overwrite.
10. Runner never executes Git/package/manifest capture itself.
11. Multi-track provenance IDs exactly match actual scenario IDs in order.
12. Provenance failures prevent env/model/learning creation.
13. Later training failure preserves the already-written immutable provenance.
14. Existing `state.json`/checkpoint/final-model semantics remain intact.
15. WP6 does not run or inspect held-out policy performance.

---

### Task 1: Define immutable provenance contracts and deterministic JSON document shape

**Files:**
- Create: `f1rl/learning/provenance.py`
- Create: `tests/learning/provenance/__init__.py`
- Create: `tests/learning/provenance/test_contract.py`
- Create: `tests/learning/provenance/test_document.py`

**Interfaces:**
- Produces immutable DTOs plus a pure document-conversion helper.
- No Git subprocess, package lookup, filesystem hashing, manifest resolution, or TrainingRunner integration yet.

- [ ] **Step 1: Write failing DTO contract tests**

Pin frozen + slots contracts equivalent to:

```python
FileFingerprint(
    role="experiment_definition",
    path="configs/experiments/ppo-generalization-v1.yml",
    sha256="a" * 64,
)

GitProvenance(
    commit_sha="b" * 40,
    dirty=False,
)

PackageVersion(name="numpy", version="2.1.0")

RuntimeProvenance(
    python_version="3.12.10",
    python_implementation="CPython",
    system="Windows",
    release="11",
    machine="AMD64",
    packages=(...),
)

TrackProvenance(
    track_id="straight-sprint-v1",
    world_sha256="c" * 64,
    content_sha256="d" * 64,
)

RunProvenance(
    schema_version=1,
    experiment_id="ppo-generalization-v1",
    protocol_version=1,
    experiment_status="draft",
    training_protocol_status="pilot_owned",
    git=...,
    files=(...),
    runtime=...,
    tracks=(...),
)
```

Pin:
- exact provenance schema version `1`;
- non-blank identifiers/names/version strings;
- SHA-256 = 64 lowercase hexadecimal characters;
- Git commit SHA = exactly 40 lowercase hexadecimal characters;
- booleans are real bools;
- positive integer protocol version with bool rejected;
- experiment status only `draft|frozen`;
- training protocol status only `pilot_owned|frozen`;
- file roles exactly:
  - `experiment_definition`
  - `training_runtime_config`
  - `track_manifest`
- `RunProvenance.files` contains exactly those three roles once each in that fixed order;
- file paths are non-empty relative POSIX text, no backslash, absolute path, `..` escape, or duplicate path;
- runtime package tuple has exactly the fixed v1 ordered package names:
  - `numpy`
  - `gymnasium`
  - `stable-baselines3`
  - `torch`
  - `PyYAML`
- selected tracks are a non-empty tuple with unique non-blank IDs;
- all DTO fields are plain immutable scalar/tuple/nested DTO values.

- [ ] **Step 2: Write failing document-shape tests**

Define a private or public pure conversion such as:

```python
_provenance_document(provenance: RunProvenance) -> dict[str, object]
```

Pin exact JSON-facing top-level keys/order:

```text
schema_version
experiment
git
files
runtime
tracks
```

Pin exact nested keys as approved by the spec.

Require:
- no `Path` values;
- no absolute repo path;
- no timestamp;
- no run ID;
- no PID;
- no hostname/username/device name;
- file role is explicit in every file record;
- package and track order preserved.

- [ ] **Step 3: Run Task 1 tests and verify RED**

```powershell
& $python -m pytest tests/learning/provenance/test_contract.py tests/learning/provenance/test_document.py -q
```

Expected: import/collection RED because `f1rl.learning.provenance` does not exist.

- [ ] **Step 4: Implement only DTOs + pure document conversion**

Use small private validators.

Do not import:
- `subprocess`;
- `importlib.metadata`;
- track manifest loader;
- TrainingRunner;
- SB3/Torch.

Do not write files yet.

- [ ] **Step 5: Run Task 1 tests**

```powershell
& $python -m pytest tests/learning/provenance/test_contract.py tests/learning/provenance/test_document.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add f1rl/learning/provenance.py tests/learning/provenance
git commit -m "feat(provenance): define run provenance contract"
```

---

### Task 2: Implement repository-relative exact-byte file fingerprints

**Files:**
- Modify: `f1rl/learning/provenance.py`
- Create: `tests/learning/provenance/test_file_fingerprints.py`

**Interfaces:**
- Consumes repository root + canonical source file paths.
- Produces validated `FileFingerprint` values.
- No Git/runtime/manifest selection yet.

- [ ] **Step 1: Write failing SHA-256 byte-identity tests**

Using temporary files, prove:

```text
sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
```

Pin:
- same bytes -> same digest;
- one whitespace byte difference -> different digest;
- LF vs CRLF bytes -> different digest;
- empty file still has valid deterministic SHA-256;
- hashing reads binary bytes, not decoded text.

- [ ] **Step 2: Write failing repository-relative path tests**

Recommended private helper:

```python
_fingerprint_file(
    *,
    repository_root: Path,
    source: Path,
    role: str,
) -> FileFingerprint
```

Require:
- repository root must exist and be a directory;
- source must exist and be a file;
- source must resolve inside repository root;
- role must be one of the three v1 roles;
- stored path uses forward slashes;
- stored path is relative to repository root;
- symlink/path resolution escaping repo root is rejected;
- no absolute path is retained in DTO/document.

- [ ] **Step 3: Pin fixed source-role ordering helper**

Add a pure/helper operation that assembles exactly:

```text
experiment_definition
training_runtime_config
track_manifest
```

in that order regardless of filesystem traversal order.

No generic “fingerprint arbitrary directory” API.

- [ ] **Step 4: Run Task 2 tests and verify RED**

```powershell
& $python -m pytest tests/learning/provenance/test_file_fingerprints.py -q
```

Expected: RED on missing helpers/behavior.

- [ ] **Step 5: Implement minimal exact-byte fingerprint helpers**

Use:
- `Path.resolve()`;
- `Path.relative_to()`;
- `hashlib.sha256()`;
- `read_bytes()`.

Do not canonicalize YAML/JSON.

- [ ] **Step 6: Run Task 1–2 provenance tests**

```powershell
& $python -m pytest tests/learning/provenance/test_contract.py tests/learning/provenance/test_document.py tests/learning/provenance/test_file_fingerprints.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```powershell
git add f1rl/learning/provenance.py tests/learning/provenance/test_file_fingerprints.py
git commit -m "feat(provenance): fingerprint canonical input files"
```

---

### Task 3: Implement Git and runtime/package capture
**Files:**
- Modify: `f1rl/learning/provenance.py`
- Create: `tests/learning/provenance/test_git_capture.py`
- Create: `tests/learning/provenance/test_runtime_capture.py`

**Interfaces:**
- Produces `GitProvenance` and `RuntimeProvenance`.
- Still no experiment/manifest orchestration or runner integration.

- [ ] **Step 1: Write Git tests using dedicated temporary repositories**

Tests create isolated repositories via real `git init`, configure local test identity, create/commit a file, and exercise the real collector.

Recommended helper:

```python
_capture_git_provenance(repository_root: Path) -> GitProvenance
```

Pin:
- clean repo -> current 40-char HEAD + `dirty=False`;
- unstaged tracked modification -> `dirty=True`;
- staged modification -> `dirty=True`;
- untracked file -> `dirty=True`;
- ignored file alone -> remains `dirty=False`;
- repo with both ignored + real dirty content -> true;
- not-a-Git-repo -> explicit failure;
- missing/non-directory root -> explicit failure.

Do not monkeypatch Git command output for these semantic tests.

- [ ] **Step 2: Pin Git subprocess behavior**

Use fixed commands equivalent to:

```text
git -C <repo> rev-parse HEAD
git -C <repo> status --porcelain=v1 --untracked-files=all
```

Require:
- no shell string execution; pass argument vectors;
- finite timeout;
- text decoding is explicit;
- nonzero exit -> `RuntimeError`/project-owned explicit failure;
- empty/invalid HEAD output rejected;
- no silent `unknown` fallback.

The exact timeout value is implementation detail but should be modest (for example 10 seconds) and tested only structurally where useful.

- [ ] **Step 3: Write runtime capture tests**

Recommended helper:

```python
_capture_runtime_provenance(
    version_lookup: Callable[[str], str] = importlib.metadata.version,
) -> RuntimeProvenance
```

Pin standard-library platform facts:
- `platform.python_version()`;
- `platform.python_implementation()`;
- `platform.system()`;
- `platform.release()`;
- `platform.machine()`.

Pin package lookup call order exactly:

```text
numpy
gymnasium
stable-baselines3
torch
PyYAML
```

Test with a test-only callable/dedicated helper under tests so values are deterministic.

Require:
- missing package lookup raises explicit failure naming the distribution;
- blank/non-string returned version rejected;
- package order remains fixed.

- [ ] **Step 4: Audit source to prove runtime capture does not import frameworks**

Add a test reading `f1rl/learning/provenance.py` and require absence of:
- `stable_baselines3`;
- `import torch`;
- `from torch`;
- Gym environment imports;
- PPO/vector-env imports.

The literal string `"stable-baselines3"` as a distribution name is allowed; distinguish imports from metadata names in the assertion.

- [ ] **Step 5: Run Task 3 tests and verify RED**

```powershell
& $python -m pytest tests/learning/provenance/test_git_capture.py tests/learning/provenance/test_runtime_capture.py -q
```

Expected: RED before implementation.

- [ ] **Step 6: Implement Git/runtime capture**

Use only standard library.

Do not store:
- hostname;
- username;
- home path;
- environment dump;
- repository absolute path.

- [ ] **Step 7: Run all provenance tests so far**

```powershell
& $python -m pytest tests/learning/provenance -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```powershell
git add f1rl/learning/provenance.py tests/learning/provenance
git commit -m "feat(provenance): capture git and runtime state"
```

---

### Task 4: Implement canonical experiment + manifest provenance capture

**Files:**
- Modify: `f1rl/learning/provenance.py`
- Create: `tests/learning/provenance/test_capture.py`
- Create or modify only test fixtures/helpers under `tests/learning/provenance/**`.

**Interfaces:**
- Consumes repository root, experiment YAML path, loaded `ExperimentDefinition`, explicit selected track IDs.
- Produces complete immutable `RunProvenance`.
- Does not persist JSON yet.

- [ ] **Step 1: Write failing argument/reload consistency tests**

Public signature:

```python
capture_run_provenance(
    *,
    repository_root: str | Path,
    experiment_path: str | Path,
    definition: ExperimentDefinition,
    selected_track_ids: tuple[str, ...],
) -> RunProvenance
```

Pin:
- repository_root accepts `str|Path`, rejects blank/non-path values;
- experiment_path accepts `str|Path`, rejects blank/non-file/outside-repo values;
- definition must be `ExperimentDefinition`;
- selected IDs must be non-empty tuple, unique, non-blank strings;
- reload `ExperimentDefinition.from_yaml(experiment_path)`;
- if reloaded value != supplied definition, fail before returning provenance.

This guards callers training from one in-memory definition while fingerprinting another file.

- [ ] **Step 2: Pin canonical source-file fingerprints**

Using a temporary repo layout or controlled copy of the real config references, verify capture includes exactly:
1. experiment YAML exact-byte SHA;
2. `definition.training.runtime_config` exact-byte SHA;
3. `definition.tracks.manifest` exact-byte SHA.

Require all three referenced paths remain within `repository_root`.

Do not fingerprint arbitrary other configs.

- [ ] **Step 3: Write selected-track manifest tests**

Use the existing `load_track_manifest()` path.

Pin:
- selected IDs resolve hashes from verified manifest entries;
- selected caller order is preserved;
- missing selected ID fails explicitly;
- duplicate selection fails before result creation;
- manifest tampering continues to fail through existing loader behavior;
- no asset hash is independently recomputed by provenance.

Use training IDs for real project integration evidence.

Do not execute track environments or policies.

- [ ] **Step 4: Pin experiment lifecycle fields**

For the current canonical project definition, prove capture records honestly:

```text
experiment_id = ppo-generalization-v1
protocol_version = 1
experiment_status = draft
training_protocol_status = pilot_owned
```

Do not call `require_frozen()` inside WP6 capture.

- [ ] **Step 5: Pin deterministic equality under injected stable runtime/Git facts**

Where private seams are needed, use legitimate helper-level injection/monkeypatch from tests to provide stable Git/runtime values.

Two captures with unchanged:
- file bytes;
- definition;
- selected IDs;
- Git/runtime facts;

must compare equal.

Do not add timestamp/run ID to make captures unique.

- [ ] **Step 6: Run Task 4 tests and verify RED**

```powershell
& $python -m pytest tests/learning/provenance/test_capture.py -q
```

Expected: RED before public capture exists.

- [ ] **Step 7: Implement `capture_run_provenance` minimally**

Flow:
1. normalize/validate repo root;
2. normalize/validate experiment path inside repo;
3. reload definition and compare;
4. verify referenced config/manifest files inside repo;
5. fingerprint exact three files;
6. capture Git;
7. capture runtime versions;
8. load verified manifest;
9. resolve selected track entries preserving order;
10. construct `RunProvenance`.

Do not mutate source files.

- [ ] **Step 8: Run provenance suite**

```powershell
& $python -m pytest tests/learning/provenance -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```powershell
git add f1rl/learning/provenance.py tests/learning/provenance
git commit -m "feat(provenance): capture canonical run identity"
```

---

### Task 5: Implement deterministic immutable provenance JSON writer
**Files:**
- Modify: `f1rl/learning/provenance.py`
- Create: `tests/learning/provenance/test_writer.py`

**Interfaces:**
- Consumes a completed `RunProvenance`.
- Writes one immutable UTF-8 `provenance.json`-style file.
- Does not integrate TrainingRunner yet.

- [ ] **Step 1: Write failing writer tests**

Public signature:

```python
write_run_provenance(
    path: str | Path,
    provenance: RunProvenance,
) -> None
```

Pin:
- valid existing parent directory required;
- destination must not already exist;
- path pointing to directory rejected;
- provenance type required;
- UTF-8 text;
- `json.dumps(..., indent=2)`-style stable formatting with one trailing newline;
- exact document content equals pure `_provenance_document()`;
- repeated serialization of same snapshot produces byte-identical output when written to different paths;
- no absolute repo path leaks into bytes;
- no NaN/Infinity.

- [ ] **Step 2: Pin no-overwrite immutability**

Create destination with sentinel content.

Calling writer must:
- raise `FileExistsError` (or clearly pinned project exception);
- preserve sentinel bytes exactly;
- create no temporary replacement artifact.

There is no update/overwrite flag.

- [ ] **Step 3: Pin failed-write cleanup**

If implementation uses atomic temp + `os.replace`, note that replace would conflict with no-overwrite semantics unless destination nonexistence is rechecked safely.

Prefer the simplest course-project-safe implementation:
- exclusive create mode (`"x"`) for destination;
- write complete deterministic JSON;
- on write failure after partial creation, best-effort delete the newly created file and re-raise.

Tests should monkeypatch/induce write failure only to verify:
- original exception remains visible;
- incomplete new provenance file is not left behind.

Do not introduce a generic artifact-store abstraction.

- [ ] **Step 4: Implement writer**

Use plain JSON-compatible document values.

Do not add a reader/parser API unless implementation truly needs one; WP6 writes snapshots and ordinary JSON tools can inspect them.

- [ ] **Step 5: Run Task 5 tests**

```powershell
& $python -m pytest tests/learning/provenance/test_writer.py tests/learning/provenance -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```powershell
git add f1rl/learning/provenance.py tests/learning/provenance/test_writer.py
git commit -m "feat(provenance): write immutable provenance snapshots"
```

---

### Task 6: Integrate supplied provenance with TrainingRunner

**Files:**
- Modify: `f1rl/learning/training/runner.py`
- Modify: `f1rl/learning/training/__init__.py` only if required by changed result export.
- Modify: `tests/learning/training/test_training_runner.py`
- Create: `tests/learning/training/mocks/mock_provenance.py` only if a reusable test helper is actually needed.
- Modify/create integration coverage:
  - `tests/integration/test_training_orchestration_integration.py`

**Interfaces:**
- `TrainingRunner(config, provenance=None)`
- optional `TrainingRunResult.provenance_path: Path | None`
- state gains exact key `provenance_file`.

- [ ] **Step 1: Write failing constructor/result/state compatibility tests**

Pin:
- `TrainingRunner(config)` remains valid;
- `TrainingRunner(config, provenance=None)` behaves identically;
- wrong provenance type rejected at construction;
- `TrainingRunResult` stays frozen and gains `provenance_path` only if adopted;
- initial state always contains:
  ```json
  "provenance_file": null
  ```
  when no snapshot supplied.

This intentionally changes the state schema in one explicit, tested way.

Existing state keys/values remain unchanged.

- [ ] **Step 2: Write failing supplied-provenance persistence tests**

With a valid test provenance:
- `state.json["provenance_file"] == "provenance.json"`;
- `run_directory/provenance.json` exists before env construction;
- bytes equal `write_run_provenance` output for supplied object;
- file remains unchanged through checkpoint/final-state updates;
- returned `provenance_path` equals file path if result field is adopted.

Use existing training mocks for PPO/vector behavior; do not put provenance fakes in production.

- [ ] **Step 3: Pin multi-track ID consistency before env creation**

For:

```python
runner.run_multi(scenarios)
```

when provenance is supplied require:

```text
tuple(track.track_id for track in provenance.tracks)
==
tuple(scenario.track_id for scenario in scenarios.scenarios)
```

Pin:
- exact match -> proceed;
- same set different order -> reject;
- missing/extra ID -> reject;
- rejection happens before run directory/env/model creation if possible.

The implementation should perform this consistency check before `_prepare_run` so mismatch leaves no run artifact directory.

- [ ] **Step 4: Pin single-track semantics**

Current `F1RLEnvFactory` carries no authored track identity.

Therefore for `run_single(factory)`:
- provenance may be absent and existing behavior continues;
- if provenance is supplied, runner persists it but must not pretend to verify the single factory's authored track hash/ID;
- no invented `training_tracks` ID appears in `state.json`.

Document this boundary in tests/comments; do not add track identity to `F1RLEnvFactory` in WP6.

- [ ] **Step 5: Pin provenance-write failure lifecycle**

Induce `write_run_provenance` failure after initial `state.json` creation.

Require:
- env builder not called;
- PPO model not created;
- no learning;
- state becomes `failed` if state exists;
- `provenance_file` remains the intended relative name or is pinned to `None` according to one chosen truthful semantic;
- incomplete provenance file absent;
- original write exception re-raised.

Choose and pin one truthful state semantic before implementation. Recommended:
- initial state writes `provenance_file="provenance.json"` when provenance was supplied;
- failed provenance write leaves state `failed` with that intended artifact reference even though file is absent, making attempted provenance binding visible.

- [ ] **Step 6: Pin later training failure preserves provenance**

When provenance write succeeds but model learning later fails:
- `provenance.json` remains present and byte-identical;
- state becomes `failed`;
- provenance is not deleted or modified;
- existing checkpoint/final-model failure rules stay unchanged.

- [ ] **Step 7: Implement minimal runner integration**

Recommended changes:
- add optional provenance field on `TrainingRunner`;
- extend `_RunArtifacts` with `provenance_path` only if useful;
- add `provenance_file` to initial state;
- persist supplied snapshot after initial state creation and before env/model construction;
- consistency-check multi-track IDs before artifact preparation;
- never call provenance capture from runner.

Do not import:
- `subprocess`;
- `importlib.metadata`;
- experiment loader;
- manifest loader;

inside `runner.py`.

Import only project-owned provenance DTO/writer.

- [ ] **Step 8: Run focused TrainingRunner tests**

```powershell
& $python -m pytest tests/learning/training/test_training_runner.py -q
```

Expected: PASS.

- [ ] **Step 9: Extend tiny real TrainingRunner integration**

Use the existing real CPU smoke setup but with a prepared deterministic test provenance object.

Prove one tiny run produces together:

```text
state.json
provenance.json
final model
(optional checkpoint depending existing smoke config)
```

Assert:
- no policy-quality claim;
- state references provenance;
- provenance bytes remain unchanged after completion;
- model artifact semantics stay intact.

This test may use a compact synthetic valid provenance object; it does not need real project Git capture because capture is already tested separately.

- [ ] **Step 10: Run orchestration integration/regression**

```powershell
& $python -m pytest tests/learning/training/test_training_runner.py tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit Task 6**

```powershell
git add f1rl/learning/training tests/learning/training tests/integration/test_training_orchestration_integration.py
git commit -m "feat(provenance): bind snapshots to training runs"
```

---

### Task 7: Canonical project provenance integration proof
**Files:**
- Create: `tests/integration/test_run_provenance.py`
- No production changes expected unless integration reveals a real WP6 defect.

**Interfaces:**
- Uses the actual canonical project experiment YAML and real track manifest.
- Captures **training-track identity only**.
- Does not execute PPO, environments, or held-out outcomes.

- [ ] **Step 1: Capture the current canonical experiment definition**

Load:

```text
configs/experiments/ppo-generalization-v1.yml
```

Use:

```python
selected_track_ids = definition.tracks.training_ids
```

Do not select held-out IDs in this integration test.

- [ ] **Step 2: Pin current lifecycle identity**

Assert the real capture records:

```text
experiment_id = ppo-generalization-v1
protocol_version = 1
experiment_status = draft
training_protocol_status = pilot_owned
```

This is not a claim that the final scientific protocol is frozen.

- [ ] **Step 3: Pin source file roles/paths**

Require exact repository-relative paths:

```text
configs/experiments/ppo-generalization-v1.yml
config.yml
assets/tracks/manifest.json
```

Compute expected exact-byte SHA values inside the test from those files and compare to capture.

Do not hard-code the digest text in the test unless there is a strong reason; the purpose is to prove correct source binding rather than freeze whitespace forever.

- [ ] **Step 4: Pin the four training track hashes**

Require captured track IDs exactly:

```text
straight-sprint-v1
slalom-v1
right-turn-v1
ramp-v1
```

and compare `world_sha256/content_sha256` to the verified real manifest entries.

No environment is created.

No held-out asset is loaded/executed beyond any manifest integrity behavior already intrinsic to `load_track_manifest`.

Important: if the existing manifest loader verifies every manifest entry including held-out asset file integrity, that is allowed because it reads identity/integrity only; the test must not instantiate held-out episodes or inspect performance.

- [ ] **Step 5: Pin actual runtime package capture**

Assert all five package records exist, are non-blank, and match `importlib.metadata.version(name)`.

Do not assert GPU/CUDA availability.

- [ ] **Step 6: Pin Git capture against current test checkout carefully**

Because the worktree will be dirty while the new integration test exists before commit, do **not** hard-code `dirty=False` inside the test.

Instead:
- assert captured commit SHA equals `git rev-parse HEAD`;
- independently derive expected dirty from porcelain and compare.

After final commit/clean branch, whole-WP6 verification should separately prove actual final branch `dirty=False`.

- [ ] **Step 7: Write deterministic snapshot to a temp destination**

Assert:
- document contains no absolute project path;
- two writes from the same captured value to separate temp destinations are byte-identical;
- JSON parses successfully and has schema version 1.

- [ ] **Step 8: Run integration test**

```powershell
& $python -m pytest tests/integration/test_run_provenance.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 7**

```powershell
git add tests/integration/test_run_provenance.py
git commit -m "test(provenance): verify canonical run identity"
```

---

### Task 8: Whole-WP6 audit and exact verification

**Files:**
- Modify only if review exposes a real Critical/Important defect.
- No feature expansion.

**Interfaces:**
- Consumes all WP6 commits.
- Produces exact verification evidence and a clean branch ready for merge decision.

- [ ] **Step 1: Verify changed production scope**

From base `8d1c2a7bd2ed22e01c580063f63806f2b4b17807`, production changes should be limited to:

```text
f1rl/learning/provenance.py
f1rl/learning/training/runner.py
f1rl/learning/training/__init__.py   # only if actually needed
```

No diff permitted under:
- `f1rl/sandbox/**`;
- `f1rl/runtime/**`;
- `f1rl/track_assets/**`;
- `f1rl/learning/gym_environment.py`;
- `f1rl/learning/observation.py`;
- `f1rl/learning/action.py`;
- `f1rl/learning/rewards/**`;
- `f1rl/learning/episode/**`;
- `f1rl/learning/algorithms/**`;
- `f1rl/learning/training/vector_env.py`;
- `f1rl/learning/evaluation/**`;
- `assets/tracks/**`;
- `configs/experiments/ppo-generalization-v1.yml`;
- `config.yml`.

- [ ] **Step 2: Audit provenance privacy boundary**

Search production/tests and review document output to require no provenance field for:
- username;
- hostname;
- home path;
- absolute repository path;
- IP/MAC;
- device ID/serial;
- environment variable dump.

Allowed platform facts remain only:
- Python version/implementation;
- OS system/release;
- machine architecture;
- package versions.

- [ ] **Step 3: Audit Git/dirty semantics**

Review exact implementation and tests:
- HEAD captured with real Git;
- porcelain includes untracked non-ignored files;
- ignored-only does not dirty;
- no patch/blob capture;
- no claim that `dirty=True` is reproducible;
- no silent fallback.

- [ ] **Step 4: Audit source/track identity semantics**

Require:
- config/manifest files use exact byte SHA;
- paths are repo-relative POSIX;
- no semantic YAML serializer invented;
- selected track hashes come from verified manifest;
- order preserved;
- no track geometry/hash duplication in WP6.

- [ ] **Step 5: Audit TrainingRunner ownership**

Require:
- runner does not run Git;
- runner does not import `importlib.metadata`;
- runner does not load ExperimentDefinition;
- runner does not load manifest;
- runner does not recompute hashes;
- runner only validates supplied IDs, writes supplied snapshot, references file.

- [ ] **Step 6: Audit held-out discipline**

Require:
- WP6 source contains no policy/environment execution;
- canonical integration selects only `definition.tracks.training_ids`;
- no held-out performance metric/result exists;
- no protocol/training tuning occurs.

Reading manifest identity/integrity for all entries through existing manifest loader is acceptable and must not be misclassified as held-out outcome inspection.

- [ ] **Step 7: Run focused WP6 tests**

```powershell
& $python -m pytest tests/learning/provenance tests/learning/training/test_training_runner.py tests/integration/test_run_provenance.py tests/integration/test_training_orchestration_integration.py -q
```

Expected: PASS.

- [ ] **Step 8: Run learning/orchestration/evaluation regressions**

```powershell
& $python -m pytest tests/learning tests/integration/test_training_orchestration_integration.py tests/integration/test_vectorized_ppo_integration.py tests/integration/test_evaluator_metrics.py -q
```

Expected: PASS.

- [ ] **Step 9: Run full/static/dependency gate**

```powershell
& $python -m pytest -q -rs
& $python -m compileall -q f1rl scripts
& $python -m pip check
git diff --check 8d1c2a7bd2ed22e01c580063f63806f2b4b17807...HEAD
```

Production dependency audit must prove:
- SB3 import owners remain exactly `ppo.py` and `vector_env.py`;
- direct production Torch imports remain zero;
- `provenance.py` does not import SB3/Torch/Gym environment/PPO/vector env;
- production Mock/MagicMock/Fake markers remain zero;
- sandbox/runtime/track_assets do not import provenance upward.

- [ ] **Step 10: Verify final project worktree Git capture**

On exact committed final WP6 HEAD, with worktree clean:

```powershell
git status --porcelain=v1 --untracked-files=all
git rev-parse HEAD
```

Then invoke the real provenance collector against the canonical experiment/training IDs and verify:
- `git.commit_sha == final HEAD`;
- `git.dirty == false`.

This is verification evidence, not a committed fixture value.

If ignored/generated files exist, they must not affect dirty status.

- [ ] **Step 11: Whole-WP6 review**

Review base...HEAD with priority:
1. privacy leak through paths/machine metadata;
2. dirty-state under-reporting;
3. package capture importing frameworks;
4. wrong config file fingerprints;
5. caller definition/file mismatch not detected;
6. wrong selected track hashes/order;
7. mutable/overwriteable provenance file;
8. runner captures facts itself;
9. provenance write occurs after training starts;
10. multi-track mismatch creates env/model or misleading artifacts;
11. training failure deletes/changes provenance;
12. changes to training policy/physics/evaluation behavior;
13. accidental WP7 scope.

Any Critical/Important defect gets a regression test before repair.

- [ ] **Step 12: Exact-final rerun after any review fix**

Repeat Steps 7–10 on the exact final HEAD.

Record:
- final HEAD;
- commit list;
- focused test count;
- learning/regression count;
- full test count + skip reason;
- compileall result;
- pip check result;
- diff-check result;
- dependency/import audit;
- privacy audit;
- canonical file/track provenance proof;
- final real Git collector result (`dirty=False`);
- final `git status`.

- [ ] **Step 13: Final branch state and stop**

Require:

```text
branch == research/run-provenance
git status --porcelain == empty
all verification commands green
canonical final provenance capture says dirty == false
no merge/push
WP7 not started
no held-out outcome inspected
```

Stop at integration gate and ask the user whether to merge, push, or keep.

## Stop Condition

After Task 8, **STOP**. Do not begin WP7 Pilot Training automatically.
