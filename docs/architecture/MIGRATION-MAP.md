# Migration map

The package was moved directly from `f1rl/` to `src/`; no `f1rl` alias is installed. Baseline: `a909a251083d937bce23e229111d0c1c1a9947f3`.

| Baseline | Current responsibility |
| --- | --- |
| `f1rl/runtime` | `src/core`: simulation sessions and physics adapters |
| `f1rl/sandbox` | `src/simulation`: deterministic world, geometry and physics |
| `f1rl/track_assets` | `src/tracks`: track assets and persistence |
| `f1rl/learning` | `src/rl`: Gym, PPO interface, action, observation and rewards |
| `f1rl/learning/training` | `src/training`: orchestration and training configuration |
| `f1rl/learning/evaluation` | `src/evaluation`: episode evaluation and metrics |
| `f1rl/research` | `src/validation`: evidence and research gates |
| `f1rl/local_demo` | `src/interfaces`: registered campaign execution, admission, evaluation, runtime selection and inference adapters |
| `f1rl/exploratory` | `src/exploratory`: restricted exploratory execution |

The original names remain inside immutable historical records and the WP7 historical audit-only allow-list. Current source references in the software-only readiness matrix and newly generated source-provenance reports point to current files. Scientific schema identifiers and frozen evaluator and RNG descriptors retain their original values. Training YAML parsing was consolidated in `src/training/config_loading.py`; malformed input and path resolution continue to be verified by the existing configuration tests. README-only placeholders for an unimplemented API and driver adapter were consolidated into `src/README.md`; no UI–Python bridge was built.

## WP3 campaign and evaluation responsibilities

The public entrypoints remain `src.interfaces.campaign.run_campaign` and `src.interfaces.evaluation.run_evaluation`; their supporting work is divided without changing artifact formats:

| Module | Responsibility |
| --- | --- |
| `src/interfaces/campaign_contract.py` | Frozen campaign config, schedule, preflight and manifest construction |
| `src/interfaces/_campaign_validation.py` | Per-run telemetry, integrity and checkpoint validation |
| `src/interfaces/_campaign_reports.py` | Campaign reports, deliverables and final evidence inventory |
| `src/interfaces/_evaluation_roster.py` | Paired deterministic evaluation seed roster |
| `src/interfaces/_evaluation_events.py` | Append-only evaluation lifecycle records |
| `src/interfaces/_evaluation_reports.py` | Idempotent evaluation text and SVG reports |
| `src/interfaces/_artifacts.py` | Shared canonical JSON, hashing, strict reads and safe writes |

`campaign.py` and `evaluation.py` remain compatibility-facing modules. Evaluation no longer imports Campaign. Campaign creates the roster through the light-weight roster module and lazily calls evaluation after training; the dependency is one-way.

## WP7 protocol and runtime capture

| Module | Responsibility |
| --- | --- |
| src/validation/wp7_protocol.py | Compatibility facade, canonical protocol digest, current source binding, seed binding, runtime-manifest validation |
| src/validation/wp7_registration.py | Registration path/profile lookup, exact schema and registration validation |
| src/validation/wp7_scenarios.py | Registered TRAIN scenario construction; preparation only, no admission |
| src/validation/wp7_runtime.py | Current host fingerprint, GPU probe and explicit model/checkpoint ZIP hashes |
| src/validation/wp7_admission.py | Admission composition; imports the public runtime fingerprint API |

`wp7_protocol.py` retains the compatibility API, canonical digest implementation, source binding, seed binding and runtime-manifest entrypoints. Canonical digest has **not** yet been extracted to its own responsibility module: doing so changes source bytes included in WP7 protocol binding and changes future protocol digests and derived evaluation seeds. This is tracked as an open Phase 5 item pending an approved provenance-compatible migration plan. The facade retains the historical runtime seam used by restore callers. Runtime capture is descriptive and does not grant training permission. Historical manifests and evidence remain immutable.

See [Deep Source Refactor Final Audit](FINAL-DEEP-SOURCE-REFACTOR-AUDIT.md) for the current phase status and gate results.

## PPO lifecycle and inference status

src.training.TrainingRunner and src.interfaces.runner remain separate orchestration owners because generic registered runs and WP3 campaign runs have different artifact, admission, telemetry and failure-state contracts. The identical chunked SHA-256 file helper is shared through src/interfaces/_artifacts.py; runner orchestration is not combined.

src.evaluation.rollout.evaluate_policy_episode is the seeded read-only rollout row API. The inference CLI and exploratory pilot use it; the CLI sends progress through an observer and does not patch env.step. The pilot adapter preserves its existing policy row format and invalid-sample handling.
