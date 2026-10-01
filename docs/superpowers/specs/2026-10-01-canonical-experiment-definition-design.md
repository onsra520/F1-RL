# Canonical Experiment Definition Design

**Status:** Approved conversational design; written specification for review.

**Branch:** `models/ppo-baseline`

**Base HEAD at spec creation:** `f9b56f5d71b41d28b70db160135ae2755ede827c`

## Goal

Define one canonical, machine-readable experiment contract for the research question:

> **How well does a PPO navigation agent generalize to unseen user-generated racing tracks?**

The contract must separate scientific protocol configuration from runtime/training configuration and track data, prevent test fixtures from becoming accidental research defaults, identify which experiment variables are fixed, explicitly represent values that are not yet scientifically justified, support a clear `draft -> frozen` lifecycle, and support protocol versioning when frozen scientific assumptions change.

WP1 does not yet select arbitrary numeric values for environment parameters when the repository has no canonical scientific source for those values.

## Existing repository state

The repository already contains validated immutable contracts for `VehicleParameters`, `SimulationConfig`, `RangefinderSpec`, `SensorSuite`, `ObservationConfig`, `RewardComposer`, `TargetProgressReward`, `CollisionPenalty`, `TimePenalty`, `EpisodeConfig`, `PPOConfig`, `VectorEnvConfig`, and `TrainingRunConfig`.

Canonical runtime configuration already exists for PPO hyperparameters, vector-environment configuration, and orchestration defaults through production dataclasses and checked-in YAML such as `config.yml`, `configs/training/ppo-baseline.yml`, and `configs/training/vectorized-ppo.yml`.

Canonical research numeric values do **not** yet exist for vehicle dynamics, simulation timestep/surface-query distance, rangefinder layout and range, observation normalization scales, reward term weights, or episode horizon. Concrete values for these currently appear mainly in tests and benchmarking helpers.

Examples such as `VehicleParameters(2.5, 0.3, 0.6, ...)`, `SimulationConfig(0.1, 0.5)`, `ObservationConfig(..., 20.0, 100.0)`, rangefinder distances of 10/20/30, reward weights such as 1/5/0.1, and `EpisodeConfig(100)` are therefore test or diagnostic fixtures, not approved scientific defaults. WP1 must never promote them automatically.

## Architectural decision

Use one canonical experiment artifact with an explicit lifecycle:

```text
draft
  -> scientific choices completed
  -> frozen
```

Recommended path:

```text
configs/experiments/ppo-generalization-v1.yml
```

A draft experiment may contain unresolved research-critical values. A frozen experiment may not.

The project-owned API should conceptually expose:

```python
definition = ExperimentDefinition.from_yaml(path)
definition.validate()
definition.require_frozen()
```

`validate()` checks schema, type, range, and reference correctness. `require_frozen()` additionally rejects any unresolved scientific field. There must be no silent fallback from an unresolved experiment field to a test fixture or hidden production default.

## Configuration ownership

The canonical experiment definition owns the scientific contract.

### Experiment-owned fields

The frozen experiment eventually fixes:

- experiment identity and protocol version;
- vehicle dynamics;
- simulation parameters;
- rangefinder layout, order, and ranges;
- observation normalization scales;
- reward composition and weights;
- episode horizon;
- algorithm/training-profile reference;
- seed protocol;
- track-manifest reference;
- evaluation semantics required for comparability.

### Track-owned fields

WP1 does not own road/surface geometry, obstacles, checkpoints, finish lines, boost zones, slow zones, score pickups, track-specific spawn pose, or individual track IDs. Those belong to WP2. WP1 references a track manifest only.

### Runtime/provenance-owned fields

WP1 does not own runtime facts such as Git SHA, dirty working-tree state, installed Python/SB3/PyTorch/NumPy versions, OS/platform, or resolved track hashes at execution time. Those belong to WP6 provenance.

## Candidate YAML contract

The initial draft artifact should resemble:

```yaml
experiment:
  id: ppo-generalization-v1
  protocol_version: 1
  status: draft

environment:
  vehicle:
    wheelbase: null
    ride_height: null
    max_steering_angle: null
    max_acceleration: null
    max_braking: null
    rolling_drag: null
    air_drag: null
    max_ground_speed: null
    gravity: null

  simulation:
    fixed_dt: null
    max_surface_distance: null

  observation:
    velocity_scale: null
    target_distance_scale: null
    rangefinders: []

  reward:
    target_progress:
      weight_per_meter: null
    collision:
      penalty: null
    time:
      penalty_per_second: null

  episode:
    max_steps: null

training:
  runtime_config: ../../config.yml
  protocol_status: pilot_owned

seeds:
  ppo: []
  sampling: []

tracks:
  manifest: ../../assets/tracks/manifest.json

evaluation:
  deterministic_policy: true
```

The exact implementation may refine field names while preserving this ownership and meaning.

## Frozen semantic contracts

Several semantics are already established by the implemented architecture and should be represented as fixed protocol rules rather than rediscovered later.

### Observation semantics

Policy observation remains:

```text
N rangefinders
+ local velocity xyz
+ local target direction xyz
+ target distance
+ target_available
+ grounded
```

Rangefinder order is declaration order. Vehicle local frame is `+x right`, `+y up`, `+z forward`.

The policy must not receive absolute world position, track ID, reward, previous action, privileged geometry, or evaluation telemetry.

### Action semantics

Exactly:

```text
steering in [-1, 1]
throttle in [0, 1]
brake = 0
```

### Reward semantics

The baseline reward composition is `TargetProgressReward`, `CollisionPenalty`, and `TimePenalty`. The same pre-step navigation target is used for progress-reward measurement. No checkpoint bonus, finish bonus, or game-score reward is introduced by WP1.

### Race semantics

The environment contract remains single lap, ordered checkpoints, one finish, with zero checkpoints permitted by the lower-level environment contract.

### Policy architecture

The baseline remains Stable-Baselines3 PPO with `MlpPolicy`. WP1 does not introduce algorithm selection or policy-network tuning.

## Values deliberately unresolved in WP1A

The following numeric fields remain unresolved until sufficient project evidence exists:

- vehicle parameter values;
- `fixed_dt`;
- `max_surface_distance`;
- rangefinder count;
- rangefinder directions;
- rangefinder maximum distances;
- `velocity_scale`;
- `target_distance_scale`;
- reward term weights;
- `max_steps`;
- final training timestep budget;
- final vector backend;
- final vector count;
- final seed replicate set.

These must not be inferred from current tests.

Track scale does not yet exist as authored data. Consequently, values such as sensor range, target-distance normalization, and episode horizon cannot yet be justified relative to actual course geometry. The final training budget/backend also requires WP7 pilot evidence rather than framework defaults.

## WP1A / WP1B lifecycle

WP1 is intentionally completed in two stages.

### WP1A — Contract definition

WP1A occurs first and defines the canonical artifact, schema, ownership, draft/frozen lifecycle, strict validation, versioning rules, and unresolved-field semantics. It does not claim the experiment is ready for final execution.

### WP1B — Environment protocol freeze

WP1B occurs after WP2 establishes the actual authored-track asset contract and representative geometry scale. WP1B resolves and freezes vehicle parameters, simulation parameters, sensor layout/ranges, observation normalization, reward weights, and episode horizon. This should not require changing the schema created by WP1A.

## Relationship with WP2

WP2 defines what a track asset is. It will provide the track JSON schema, canonical loader, canonical initial-world-state derivation, track ID, canonical serialization, track content fingerprint/hash, and manifest structure.

WP1 must not duplicate track geometry. It references:

```text
assets/tracks/manifest.json
```

The manifest becomes concrete during WP2.

## Relationship with WP3

WP3 owns training/held-out separation. It validates:

```text
training IDs intersect held-out IDs = empty
training hashes intersect held-out hashes = empty
```

WP3 also verifies that all scenarios share the frozen WP1 non-track contract. WP1 therefore defines what must remain equal; WP3 enforces equality across the split.

## Relationship with WP4/WP5

WP4 owns evaluation-only telemetry. WP5 owns the evaluator and metric calculation.

WP1 may fix high-level evaluation semantics necessary for reproducibility, including deterministic policy evaluation and later the approved primary metric names. It must not implement telemetry or evaluator logic.

## Relationship with WP6

WP1 states what experiment should run. WP6 records what actually ran.

WP6 later snapshots experiment ID/version, config hashes, Git SHA, dirty status, dependency versions, platform, and track IDs/hashes. No provenance collection belongs in WP1.

## Relationship with WP7

Root `config.yml` currently represents editable runtime defaults. It must not become the final scientific source of truth merely because it exists.

During development, a draft experiment may reference `../../config.yml`. WP7 pilot determines the final training timestep budget, vector backend, vector count, checkpoint policy, and any justified PPO adjustments.

After pilot, the final training configuration should be frozen as a stable experiment-owned profile, for example:

```text
configs/experiments/ppo-generalization-v1-training.yml
```

The frozen experiment then references that stable profile instead of the mutable root runtime config.

## Seed protocol

Generic APIs permit valid 32-bit unsigned seeds:

```text
0 <= seed <= 2**32 - 1
```

PPO configuration may technically support `None`, but the final scientific experiment should use explicit integer seeds.

WP1A defines the representation but does not choose arbitrary final values. The final experiment should pair one PPO seed with one sampling seed for each independent training replicate.

At least three independently trained-model replicates remain the expected baseline for WP8 unless later compute constraints justify a documented change.

## Protocol versioning

A frozen protocol is immutable in scientific meaning.

After `status: frozen`, changing any material scientific field requires a new protocol version. Examples include vehicle dynamics, simulation timestep, sensor layout/range, observation normalization, reward composition/weights, action semantics, episode horizon, PPO hyperparameters after final freeze, training budget after final freeze, train/held-out split, and evaluation semantics or primary-metric definitions.

A materially changed protocol should become, for example:

```yaml
experiment:
  id: ppo-generalization-v2
  protocol_version: 2
```

Frozen v1 must not be silently edited after observing held-out results.

## Drift prevention

Research-critical values must be explicit in the resolved frozen experiment.

A frozen definition must not rely on dataclass numeric defaults that are not recorded in the experiment, values imported from tests, implicit environment variables, mutable global configuration, or frontend/editor defaults.

Referenced configuration files are part of the resolved protocol and will later be fingerprinted by WP6. Unknown YAML keys must be rejected. Missing required frozen values must fail before training starts.

## Scope

### In scope for WP1

- canonical experiment-definition artifact;
- strict YAML schema/loader;
- immutable experiment DTOs;
- draft/frozen lifecycle;
- unresolved-field handling;
- ownership boundary;
- config-reference resolution;
- protocol-version rules;
- focused validation tests;
- documentation.

### Out of scope

- track serialization/loading;
- authored track creation;
- train/held-out split implementation;
- evaluation telemetry;
- evaluator implementation;
- provenance collection;
- real pilot training;
- actual baseline training;
- reward ablation;
- frontend/editor work.

## Acceptance requirements

WP1 implementation must prove:

1. the canonical YAML loads into immutable project-owned values;
2. unknown root and nested keys are rejected;
3. malformed types and invalid enum/status values are rejected;
4. relative references resolve from the experiment file directory;
5. draft definitions can represent unresolved research fields explicitly;
6. unresolved fields never acquire hidden defaults;
7. `require_frozen()` rejects any unresolved research-critical field;
8. frozen definitions reject `None`, empty seed protocols, and draft-only placeholders;
9. semantic baseline identifiers remain explicit;
10. no test fixture is imported or referenced by production experiment code;
11. no SB3 import boundary is expanded;
12. no direct production PyTorch import is introduced;
13. no production mock/fake/test-only path is introduced;
14. existing tests remain green.

WP1 implementation remains configuration/framework work only. It does not claim that any selected numeric protocol is scientifically optimal.

## Planned sequence after approval

```text
WP1A specification
-> WP1A implementation
-> WP2 Track Asset Contract
-> WP1B environment numeric freeze
-> WP3 Train/Held-out Split
-> WP4 Evaluation Telemetry
-> WP5 Evaluator + Metrics
-> WP6 Provenance
-> WP7 Pilot Training
-> final training-profile freeze
-> WP8 Baseline Generalization Experiment
```

This sequencing keeps the scientific protocol explicit without inventing values before the project has enough evidence to justify them.
