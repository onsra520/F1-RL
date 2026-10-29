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
