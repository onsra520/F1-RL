# Migration map

Baseline `a909a251083d937bce23e229111d0c1c1a9947f3`. Intended root change: `f1rl/` → `src/` (no `src/f1rl`), with imports, module strings, entry points, packaging and tests updated in validated batches. Domain reorganization remains to be determined from source contracts; no old-to-new module mapping is yet approved.

| Baseline | Intended responsibility | Risk |
| --- | --- | --- |
| `f1rl/sandbox`, `f1rl/runtime` | simulation and geometry | source-bound physics manifests, roll/crash terminal semantics |
| `f1rl/track_assets`, sandbox track | tracks | path fingerprints and authored asset loading |
| `f1rl/learning` | RL, training, evaluation | Gym contracts, PPO and serialized module references |
| `f1rl/research`, `f1rl/exploratory` | validation and orchestration | registered source identities, fail-closed gates and signatures |
| `f1rl/local_demo`, `scripts` | interfaces and entry points | subprocess module strings, working-directory and evidence paths |
| `web` | frontend | independent app, Python bridge not implemented |
| `docs/research/evidence` | immutable evidence | preserve paths, bytes and hashes |

A move-only experiment in an independent detached clone made the existing `source_bound_acceptance` fail on missing `f1rl/sandbox/spatial_contact_dynamics.py`. A compatible mapping must prove that source hashes still bind the actual migrated implementation, while historical keys, schemas, receipts and hashes retain their recorded meaning. No interpretation of a historical source identity as a new one may be introduced silently.
