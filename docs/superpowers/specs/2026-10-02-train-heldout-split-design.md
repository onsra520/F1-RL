# Train/Held-Out Split and Leakage Guard Design

**Status:** Written specification for review  
**Branch:** `research/train-heldout-split`  
**Base HEAD:** `a76d50f93a6c63478f507f4d653b683a9e9d6174`

## 1. Goal

WP3 turns the six WP2 candidate tracks into an explicit scientific train/held-out partition for the PPO generalization experiment.

The split must:
- live in the experiment protocol rather than the asset registry;
- make held-out membership explicit and reproducible;
- reject duplicate or missing track identities;
- reject geometry leakage even when a track is renamed or given another spawn;
- preserve the WP1B non-track environment contract across every scenario;
- keep held-out tracks unavailable for pilot tuning.

WP3 does not train PPO, define evaluation metrics, collect provenance, or freeze the whole experiment.

## 2. Research contract

The research question remains:

> How well does a PPO waypoint-guided navigation/control policy generalize to held-out track geometries?

## 3. Ownership decision

WP2 deliberately keeps `assets/tracks/manifest.json` role-free. It is an asset registry containing identity, portable path, `world_sha256`, and `content_sha256`.

WP3 therefore stores scientific split membership in the canonical experiment definition:

```yaml
tracks:
  manifest: ../../assets/tracks/manifest.json
  training_ids:
    - straight-sprint-v1
    - slalom-v1
    - right-turn-v1
    - ramp-v1
  heldout_ids:
    - left-turn-v1
    - offset-slalom-v1
```

Directory names such as `assets/tracks/training/` and `assets/tracks/evaluation/` are not authoritative. Moving an asset between those directories does not change scientific membership unless the experiment definition changes.

The manifest remains reusable by demo/editor/runtime tooling without importing research semantics.

## 4. Canonical v1 partition

The v1 split is exactly:

```text
training:
  straight-sprint-v1
  slalom-v1
  right-turn-v1
  ramp-v1

held-out:
  left-turn-v1
  offset-slalom-v1
```

This split is fixed by the experiment protocol, not inferred from paths.

The training partition exposes straight-line control, obstacle/slalom behavior, a right-angle turn, and elevation/ramp dynamics. The held-out partition then probes new geometry arrangements without withholding an entire control capability from training.

`left-turn-v1` tests directional/mirrored turn generalization from the trained right-turn case. `offset-slalom-v1` tests lateral obstacle/waypoint rearrangement relative to the training slalom case.

WP3 does not claim this six-track corpus is statistically exhaustive. It defines the controlled course-project experiment that later work will execute and report with appropriate limitations.

## 5. TrackExperimentConfig

The experiment DTO extends to:

```python
@dataclass(frozen=True, slots=True)
class TrackExperimentConfig:
    manifest: Path
    training_ids: tuple[str, ...]
    heldout_ids: tuple[str, ...]
```

Structural validation requires:
- `manifest` is a `Path`;
- `training_ids` and `heldout_ids` are tuples;
- every ID is a non-blank string;
- each partition is non-empty;
- IDs are unique within each partition;
- `training_ids ∩ heldout_ids = ∅`.

The generic DTO does not hard-code “4 training + 2 held-out”. That exact count belongs to the checked-in canonical v1 profile and its regression tests.

Declaration order is preserved. The order may later be used for deterministic reporting, but it does not imply training sampling weights.

## 6. Split resolution

WP3 adds a project-owned resolver conceptually equivalent to:

```python
resolve_track_split(config: TrackExperimentConfig) -> ResolvedTrackSplit
```

Resolved output is immutable:

```python
@dataclass(frozen=True, slots=True)
class ResolvedTrackSplit:
    training: tuple[TrackManifestEntry, ...]
    heldout: tuple[TrackManifestEntry, ...]
```

Resolution loads the referenced manifest with existing `load_track_manifest()`, so file existence, ID/path integrity, and both semantic hashes are already verified by WP2.

The resolver then:
1. resolves each declared training ID to exactly one manifest entry;
2. resolves each declared held-out ID to exactly one manifest entry;
3. rejects any declared ID absent from the manifest;
4. preserves experiment declaration order;
5. applies cross-partition leakage checks.

Manifest entries not named by either partition remain legal. This allows future demo or auxiliary assets to coexist in the registry without entering the scientific experiment.

## 7. Leakage identity

The primary leakage identity is `world_sha256`, not `track_id` and not `content_sha256`.

WP2 intentionally defines `world_sha256` over validated world geometry/entities while excluding:
- `track_id`;
- spawn pose;
- JSON formatting;
- object-key order;
- entity input order.

Therefore WP3 rejects any `world_sha256` collision across training and held-out partitions.

This catches:
- the same geometry copied under another filename;
- the same geometry with another `track_id`;
- the same geometry with another spawn;
- the same geometry with reordered or reformatted JSON.

`content_sha256` remains useful for WP6 provenance because it identifies world + spawn content. It is not sufficient for leakage prevention because changing only spawn legitimately changes `content_sha256`.

The current six-track canonical profile must also have six distinct `world_sha256` values. Tests pin that property.

## 8. Leakage scope

Required cross-partition invariant:

```text
training world hashes ∩ held-out world hashes = ∅
```

Track IDs must also be disjoint, but ID disjointness alone is not treated as scientific protection.

Within-partition duplicate world hashes are rejected as well. Repeating identical geometry under two IDs would distort exposure/reporting and falsely increase the apparent track count.

Thus every selected research track must have a globally unique `world_sha256` across the resolved split.

## 9. Non-track environment equality

All research scenarios use one canonical `ExperimentDefinition`.
Only track-owned inputs may differ per scenario:
- `TrackWorld`;
- track-specific spawn pose;
- mutable initial state derived canonically from that spawn.

The following must be identical across every training and held-out scenario:
- `VehicleParameters`;
- `SimulationConfig`;
- observation encoder and sensor suite;
- reward composer;
- episode configuration;
- action semantics;
- race semantics.

WP3 does not introduce a new generic scenario framework. Integration tests construct real `F1RLEnvFactory` values from the canonical experiment plus each resolved asset and exercise the existing `TrainingScenarioSet` equality contract.

## 10. Canonical initial-state fairness

For every selected research asset, initial mutable state is produced only through WP2 `build_initial_world_state()` using WP1B `max_surface_distance = 0.75`.

Tests verify:
- authored spawn position/orientation are preserved;
- velocity and angular velocity are zero;
- score is zero;
- collected-pickup state is empty;
- active checkpoint/finish interaction state is empty;
- initial grounded state is true for all six checked-in research tracks.

A track that cannot produce a valid canonical initial state under the frozen WP1B environment contract is not eligible for this v1 split.

## 11. Held-out usage policy
The held-out partition is a true test partition.

Before final evaluation:
- WP7 pilot/calibration may use training tracks only;
- PPO/vector/backend/budget decisions may use training tracks only;
- reward/sensor/environment tuning may use training tracks only;
- held-out episode outcomes must not be used to tune protocol v1.

WP8 may evaluate the final independently trained models on both:
- seen training tracks, to characterize in-distribution behavior;
- held-out tracks, to measure the generalization gap.

If held-out results motivate a material environment, reward, sensor, training, split, or evaluation redesign, that change belongs to protocol v2 and requires a fresh baseline. Protocol v1 is not silently retuned after held-out observation.

WP3 enforces membership and leakage identity in code. It does not attempt to prevent a human from manually opening a JSON file; the scientific process rule is documented and later experiment runners should consume the resolved split rather than ad hoc paths.

## 12. Experiment lifecycle

WP3 extends the canonical draft YAML with explicit `training_ids` and `heldout_ids`.

It does not change:
- `experiment.status: draft`;
- `training.protocol_status: pilot_owned`;
- empty PPO/sampling seed arrays;
- WP1B environment values.

`ExperimentDefinition.require_frozen()` must additionally require that the referenced split resolves successfully and passes leakage guards before a future whole-protocol freeze can succeed.

A draft experiment may call `resolve_track_split()` independently, allowing WP7 to use the training partition before final training/seeds are frozen.

## 13. Error behavior
WP3 continues the existing project convention of clear `ValueError` failures rather than adding a broad exception hierarchy.

Errors should identify the violated contract, including:
- blank/duplicate split IDs;
- overlap between training and held-out IDs;
- declared ID absent from manifest;
- duplicate selected `world_sha256`;
- manifest integrity failure inherited from WP2;
- invalid canonical initial state under the experiment environment.

No silent filtering, auto-deduplication, or fallback partition is allowed.

## 14. Public API placement

Split semantics belong with the experiment contract. The recommended module placement is:

```text
f1rl/learning/experiments/
├── definition.py
├── split.py
└── __init__.py
```

`definition.py` owns YAML/DTO structure. `split.py` owns manifest-backed partition resolution and leakage validation.

Expected exports:

```python
from f1rl.learning.experiments import (
    TrackExperimentConfig,
    ResolvedTrackSplit,
    resolve_track_split,
)
```

`f1rl.learning` itself should not begin auto-exporting experiment APIs; the existing optional-boundary behavior remains unchanged.

## 15. Dependency direction
Production dependency direction remains:

```text
learning.experiments.split -> track_assets
track_assets               -> sandbox
sandbox                    -X-> learning
track_assets               -X-> learning
```

WP3 must not expand the direct Stable-Baselines3 import boundary beyond:
- `f1rl/learning/algorithms/ppo.py`;
- `f1rl/learning/training/vector_env.py`.

WP3 introduces no direct production PyTorch imports and no production mocks/fakes.

## 16. Files expected to change

Primary implementation surface:

```text
configs/experiments/ppo-generalization-v1.yml
configs/experiments/README.md
f1rl/learning/experiments/definition.py
f1rl/learning/experiments/split.py
f1rl/learning/experiments/__init__.py
tests/learning/experiments/*
tests/integration/*
assets/tracks/README.md
```

The six track JSON files and their semantic hashes are immutable inputs to WP3 and must not change.

## 17. Required tests

### DTO/YAML
- split fields are required by the v1 experiment schema;
- unknown split keys are rejected;
- non-list YAML values are rejected before tuple conversion;
- blank IDs rejected;
- duplicates within either partition rejected;
- cross-partition ID overlap rejected;
- declaration order preserved.

### Resolver/leakage
- canonical 4+2 profile resolves;
- missing manifest ID rejected;
- same world hash across partitions rejected;
- same world hash under different IDs rejected;
- same world hash with different spawn/content hash rejected;
- duplicate world hash within one partition rejected;
- unselected manifest entries remain legal.

### Integration
- all six canonical IDs match the approved 4+2 split;
- every selected manifest entry passes existing manifest integrity checks;
- all six world hashes are distinct;
- canonical initial mutable state is fair and grounded;
- all six real factories share the WP1B non-track contract;
- existing Gym action/observation spaces remain compatible.

### Lifecycle/boundaries
- canonical experiment remains draft after WP3;
- `require_frozen()` rejects an invalid split when all earlier freeze requirements are otherwise satisfied;
- `f1rl.learning` does not auto-import the experiment API;
- SB3 direct-owner invariant remains exactly two files;
- production PyTorch imports remain zero;
- production mock/fake markers remain absent.

## 18. Non-goals

WP3 does not:
- author or modify track geometry;
- change WP1B vehicle/simulation/sensor/reward/episode values;
- choose PPO hyperparameters or final training budget;
- choose final PPO/sampling seed replicates;
- implement evaluator metrics or telemetry;
- implement runtime provenance;
- run pilot or baseline training;
- implement reward ablations;
- add a generic dataset-management framework;
- infer scientific membership from directory names.

## 19. Acceptance criteria

WP3 is complete when:
1. the canonical experiment explicitly declares the approved 4+2 partition;
2. asset manifest semantics remain role-free;
3. split DTO validation rejects malformed/overlapping membership;
4. every declared ID must exist in the verified manifest;
5. selected `world_sha256` values are globally unique across the split;
6. rename/spawn/reformat tricks cannot bypass leakage protection;
7. canonical initial-state fairness is verified across all six tracks;
8. real environment factories prove the non-track contract is identical across the split;
9. future `require_frozen()` includes split/leakage validation;
10. the experiment remains draft and pilot-owned after WP3;
11. WP2 track files/hashes remain unchanged;
12. existing dependency boundaries and full test suite remain green.

## 20. Sequence after approval

```text
WP3 written spec
→ WP3 implementation plan
→ TDD: split DTO/YAML contract
→ TDD: resolver + leakage guards
→ canonical 4+2 profile update
→ real integration/fairness tests
→ whole-WP3 review + verification
→ WP4 evaluation telemetry
```

WP3 stops before WP4.

## 21. Scientific interpretation

The split supports a controlled generalization comparison, not a claim that six tracks represent all racing geometries.

Independent trained model seeds remain the statistical replicates in WP8. Repeated deterministic evaluation of one model on one track is not an independent replicate.

The held-out set is intentionally small and fixed before pilot/baseline evaluation. Results must therefore be reported per-track and across independent model seeds, with limitations stated rather than treating the two held-out tracks as a broad population sample.
