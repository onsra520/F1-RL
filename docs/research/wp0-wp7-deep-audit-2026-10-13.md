# F1-RL — WP0–WP7 cross-package deep audit (2026-10-13)

**Baseline:** `recovery/wp7-rebuild` commit `c8d0a455a7cae1569e1d9455e1d0e8830057ae2b`.
**Scope:** user-requested audit across WP0–WP7. No work after WP7, HELD-OUT results, protocol freeze, model retraining, or merge to `main`.
**Decision:** `NOT RELEASE-READY` / `scientific_freeze_permitted=false`. All blockers must have signed-off evidence before any next research stage.

This is a risk-based code/spec/artifact audit with targeted dynamic validation, **not** a proof of zero defects or a full line-by-line formal verification. Distinguish *code-behavior confirmed*, *archive confirmed*, *spec/design mismatch*, and *unverified requirements*. Source paths below refer to the baseline commit unless noted.

## A. Evidence collected

1. Clean `recovery/wp7-rebuild` starting state; exact HEAD on remote tracked branch. `main` not modified.
2. Full local Windows .venv regression: **1,290 passed, 2 skipped**. Python `compileall` and `uv pip check` passed with 23 installed packages. This does **not** measure branch coverage, mutation score, formal correctness, or scientific validity; no `pytest-cov`, `coverage`, `hypothesis`, `ruff`, or `mypy` was installed in the audited environment.
3. Tests distributed across `tests/sandbox`, `tests/runtime`, `tests/learning`, `tests/track_assets`, and `tests/integration`. Versioned spec/plan archive exists under `docs/superpowers/{specs,plans}`. No tracked `.github/` CI workflow or lockfile was found.
4. Canonical experiment parsed as `experiment.status=draft`, `training.protocol_status=pilot_owned`, **zero final PPO/sampling seeds**. Four TRAIN IDs and two HELD-OUT metadata entries are distinct; **six unique `world_sha256`** identities. `load_track_manifest` validates all six JSON assets and canonical hashes. This is static metadata validation only, not a HELD-OUT rollout.
5. Both original exploratory final PPO runs are `completed` at **200,704 actual** timesteps, 4/4 deterministic TRAIN each, with eight verified checkpoint/evaluation milestones per run, genuine model hashes, and source fingerprints. Importantly both archived `state.json` declare `provenance_file=null`.
6. Four independent linked reports in `wp7-freeze-readiness-c8d0a45.json`: XZ continuous trajectory, conservative 3D body proxy, Right Turn centreline road-union, 128-episode TRAIN stochastic action screen. Hash/identity gate v3 returns **six blocking decisions**.
7. Original stochastic screen results: calibration **60/64** (Right Turn **12/16**), confirmation **62/64** (Slalom **15/16**, Right Turn **15/16**). Each includes a fixed Torch action seed and exact one-sided 95% binomial lower confidence bound; all-track screening **FAIL** for both models.
8. **Independent 128-episode stochastic TRAIN replay PASSED byte-for-byte reproducibility:** `wp7-train-stochastic-action-sensitivity-v1-replay-audit.json` SHA256 `ce7b86c1869aa60766eb3565f343b967c5a76285cc90a22b355fc1dc367b196b`, exactly equal to the original v1 report SHA256. Both real final PPO policies reproduce the same per-track 16-episode outcomes. This is a same-host/same-source reproducibility check, **not** independent cross-machine verification.

## B. WP-by-WP disposition

| Package | Code/spec inspected | Verified strengths | Gate / remaining work |
|---|---|---|---|
| **WP0 (undefined)** | Full tracked-code/docs search, git history | No tracked `WP0` heading or Work Package 0 contract found | **NOT AUDITABLE AS A FORMAL WP**. Define the intended WP0 charter/deliverables and acceptance criteria, or declare baseline core work as an unnumbered foundation. Do not invent a WP0 pass. |
| **WP1A — experiment contract** | `experiments/definition.py`, canonical experiment spec | Strict schema, seed uniqueness in `require_frozen()`, explicit `draft/frozen` lifecycle | **REVIEW**: canonical `_V1_SEMANTICS` only models the legacy observation/reward; WP7 reconstructed behavior cannot be honestly frozen under these old semantics without a versioned canonical schema/migration. |
| **WP1B — environment constants** | `configs/experiments/ppo-generalization-v1.yml`, vehicle physics, integrator | Fixed timestep, original numerical tuple still preserved; cross-track non-track equality checks | **BLOCKED FOR FULL-BODY CLAIMS**: original physics collision uses sampled **vehicle centre point**, with no official length/width/height/hull; no swept-body collision rule. An explicit *point-car scientific scope* may be acceptable, but cannot be claimed as full vehicle safety. |
| **WP2 — authored track assets** | `track_assets/{loader,manifest,canonical}.py`, six checked-in tracks | Strict JSON/duplicate-key/nonfinite rejection, canonical world/content hash verification, normalized paths and validated authored routes | **IMPLEMENTATION PASS within tested contract**, but measured physical passability/vehicle dimensions and sensible real-world corner radii are *not* properties certified by the JSON validator. |
| **WP3 — split/leakage protection** | `experiments/split.py`, `reconstruction.py`, `training/runner.py` | Four TRAIN IDs, two HELD-OUT IDs, distinct world hashes; WP7 canonical builder constructs **only TRAIN** environments; no HELD-OUT rollouts | **REVIEW**: generic `TrainingRunner.run_multi()` accepts a caller-supplied `TrainingScenarioSet` even without provenance and itself does not enforce the registered TRAIN/HELD-OUT split. The **WP7 entrypoint** is protected, but a future scientific command must gate identities at its own boundary. |
| **WP4 — evaluation telemetry** | `gym_environment.py`, evaluator step-info contract | Tick, elapsed time, checkpoint/finish status, events, velocity/speed, reward terms; failure handling and telemetry tests | **REVIEW**: public telemetry lacks vehicle orientation quaternion/3D body pose; `swept_volume_audit.py` must reach `env._simulation.snapshot.world.vehicle` directly. Needed for auditable future full-body evaluation. |
| **WP5 — evaluator/metrics** | `evaluation/{evaluator,metrics}.py`, checkpoint report validator | Finish and truncation consistency, reward-sum checks, progression bound, per-track success/time/reward summaries | **REVIEW**: `success` means finish under the current sampled-centre engine, not physically valid continuous vehicle trajectory. One deterministic episode/track cannot support generalization/reliability claim; per-model seeds are the independent scientific replicates. |
| **WP6 — run provenance** | `provenance.py`, `training/runner.py`, archived `state.json` | RunProvenance capture implementation and strict source/training-track equality checks exist when supplied | **HIGH BLOCKER**: `scripts/wp7_run_exploratory.py` creates `TrainingRunner` without provenance; both real 200k runs lack **training-time** `provenance.json`. Cannot reconstruct genuine host/package snapshot after the fact. Prospective integration/retention required. |
| **WP7 — training and freeze** | recovery config/entrypoint, two archived runs, 4 derived audits + gate v3 | Legitimate complete calibration and confirmation, 4/4 deterministic TRAIN each, verified hashes/seed identity, 1,290 regression pass | **BLOCKED**: action-sensitivity screen fails for both; Right Turn centre deviation 5.511796m / 4.415475m still needs approved corridor; geometry/physics/provenance/seed/backup requirements remain open. No scientific freeze; HELD-OUT sealed. |

## C. Findings register (risk-based, not arbitrary popularity ranking)

**H-01 — WP1A/WP7 canonical scientific semantic mismatch (HIGH / release gate).**
`f1rl/learning/experiments/definition.py` pins observation `rangefinders-local-navigation-v1` and reward `target-progress-collision-time-v1`; `configs/recovery/wp7-c2-candidate.yml` instead uses authored-surface route observations and route-highwater/offtrack reward. The separate recovery config avoids silent mutation, but a **versioned, freezeable canonical experiment** for the adopted C2 semantics is not present. **Required:** independently review v2 semantics/version, make `require_frozen()` enforce the new canonical mapping, maintain backward compatibility for v1, verify matching model hashes and explicit seed protocol. No retroactive relabeling.

**H-02 — WP1B/WP5/WP7 centre-point collision engine vs vehicle-safety claim (HIGH / science scope).**
`f1rl/sandbox/track/model.py::query_point` checks `volume.contains(position)`; `f1rl/sandbox/world.py` checks obstacle occupancy **only at post-integration centre position**. A swept step can cross an obstacle without generating an engine hit. **Checkpoint and finish entry events are also determined from sampled-point queries** (`world.py`, `interactions/resolver.py`); crossing a thin gate between samples is a potential missed-event class, not a demonstrated failure on the archived TRAIN episodes. Earlier audit reports use conservative body assumptions, **not actual car dimensions**; calibration large-proxy lower bound **−0.040417m (inconclusive)**. **Required:** a clearly restricted point-agent scientific claim OR a newly versioned continuous body-collision model, which would require fresh training. Do not mutate original runs.

**H-03 — WP6 provenance not integrated into archived WP7 training (HIGH / reproducibility).**
`scripts/wp7_run_exploratory.py` passes no provenance to `TrainingRunner`; both archived `state.json` have `provenance_file=null`. Training-time OS/library/Git/config chain cannot be restored exactly. SHA256 and retrospective source fingerprints are positive but insufficient to prove training-time environment. **Required:** prospective, fail-closed provenance capture before model creation; check absolute file and run-hash contract; test interrupted/crashed/restarted artifact lifecycle.

**H-04 — WP7 observed action robustness fails registered TRAIN screen (HIGH / model).**
Original 128 stochastic TRAIN episodes: calibration 60/64; confirmation 62/64. In particular Right Turn calibration 12/16, confirmation 15/16. The exact one-sided 95% Clopper–Pearson lower bounds at 12/16 and 15/16 are **0.51560358 and 0.73604260**, both less than screen threshold 0.80. Tests establish this is a fixed-geometry stochastic action sensitivity, not different-track generalization. **Required:** diagnose recorded failed TRAIN episodes; define candidate changes before retraining, independently rerun true PPO replicates and robustness on TRAIN without touching HELD-OUT.

**M-01 — WP0 not defined (MEDIUM / scope contract).**
No `WP0` / Work Package 0 contract was discovered in tracked repo/specs. Assign a real scope and acceptance evidence; otherwise mark `N/A` with explicit owner sign-off.

**M-02 — WP3 generic TrainingRunner lacks scientific split admission check (MEDIUM / misuse path).**
`TrainingRunner.run_multi()` operates on arbitrary `TrainingScenarioSet`; the canonical WP7 builder performs TRAIN-only admission but the lower-level runner allows absence of provenance. Requires a separate experiment-bound, immutable allowlist for *scientific execution*, not invasive change to a legitimate general-purpose trainer.

**M-03 — WP4 body-pose instrumentation insufficient (MEDIUM).**
The public `info.telemetry` includes vehicle position/velocity but not orientation or collision hull; 3D validator reads private simulator snapshot. Add explicit versioned read-only scientific telemetry contract before full-body evidence; do not change policy observation vector as a side effect.

**M-04 — WP7 route-corridor acceptance remains underspecified (MEDIUM / safety criterion).**
A separate Right Turn XZ check proves **0.000m centreline outside the union of two 12m roads**, but observed maximum deviation from the authored route is **5.511796m** (calibration), while conservative interpolation upper bound is **6.042907m**. The upper bound does NOT prove a >6m violation. The allowable corner-cutting threshold must be chosen by course-design intent, not tuned to make the current model pass.

**M-05 — CI/dependency reproducibility gap (MEDIUM / build).**
No tracked CI workflow, dependency lockfile, integrated coverage/property-test tools were found on this audit branch. Local Windows regression is encouraging, but it is not an independently executed multi-OS reproducibility gate. Require versioned dependency resolution and at least Windows/Linux CI checks, plus a targeted determinism/property-based suite.

**M-06 — evidence preservation and retrospective gate scope (MEDIUM).**
Original ZIP models, source and audit reports reside under `D:\Projects\.archives`, with no verified off-device checksum backup. `freeze_readiness.py` pins checksums and expected historical outcomes: suitable as a **historical record**, but not a reusable scientific release gate for a future candidate. A later clean scientific implementation must use artifact-manifest registration and independent reviewer signoff, not clone hashes or accept a mutable self-whitelist.

## D. Stop/go criteria — all required

1. WP0 scope documented (or explicitly `N/A`); WP1A canonical semantics and experiment version capable of a truthful frozen definition. Verify source and code against immutable spec.
2. WP1B physical scope resolved; if body aware, specify real shape, swept collision with step-wise geometry, contact validity, and reproducible training version. If point-agent, *no* claims of car-body collision avoidance.
3. WP2 assets and WP3 split hashes revalidated on every scientific run. No ability to substitute held-out IDs at the scientific command boundary. Exact train/heldout IDs bound into provenance.
4. WP4, WP5 complete introspectable telemetry, evidence validation and per-model statistics; independent rerun and property/metamorphic tests for physics, navigation, checkpoint ordering, reward and route metrics.
5. WP6 provenance captured *at actual run start*, immutable file hashes, package/host/Git/dirty state, verified outside-worktree durable backup + restoration test.
6. WP7 candidates registered prospectively with actual replicates, observed TRAIN robustness meeting a **prespecified** scientifically defensible criterion, valid road corridor and geometry, separate final scientific protocol freeze.
7. Only after independent approval and `require_frozen()` gate passes, decide whether to propose subsequent generalization work. Until then **zero HELD-OUT rollouts and zero post-WP7 execution**.

## E. Boundaries of this audit

- 1,290 green tests show that the present asserted behaviors match present tests, **not** that requirements are complete.
- No live video/manual driving or physical-world chassis calibration, code coverage, formal proof, external backup restore, independent cross-OS run, or independent team signoff was performed.
- No automatic fixes to physics, model, reward, split, test data, seed arrays, or frozen experiment. Changed semantics would invalidate comparability to original 200k weights.
- Dates in historic WP7 reconstruction notes preceding calibration are **snapshots**, not current evidence. The authoritative present gate is `wp7-freeze-readiness-c8d0a45.json`, which returns `review_required`, 4 linked audits, six blockers and both HELD-OUT/scientific-freeze flags false.

**Priority order:** H-01 semantics & scientific scope, H-02 physical model decision, H-03 prospective provenance, H-04 failed stochastic TRAIN reliability; then M-02 scientific split control, M-03 telemetry, M-04 corridor policy, M-05 CI, M-06 backup, M-01 WP0 charter. None of the findings warrants opening HELD-OUT as a debugging instrument.
