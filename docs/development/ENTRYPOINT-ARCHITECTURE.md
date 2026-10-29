# Script entrypoint architecture

The public versionless CLI is:

```powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
```

The registry routes 46 active stable commands to modules under functional packages. Two legacy lecturer-demo routes were retired and remain recorded in the script inventory as historical provenance. Package names and filenames are versionless except where a scientific contract identifier requires its version. The router forwards arguments and exit status; each command owns its existing admission and evidence behavior.

## Layers

| Layer | Location | Responsibility |
| --- | --- | --- |
| CLI/router | `scripts/workflows/` | Stable aliases and module dispatch |
| Executable modules | `scripts/{diagnostics,evaluation,evidence,inference,training,internal}/` | Command-line parsing and orchestration |
| Domain/application | `src/` | Reusable implementation and validation |
| Research source records | `docs/research/` | Historical scientific identity and immutable receipts |

Executables are invoked with `python -m scripts.<package>.<module>`. No executable modifies `sys.path` or imports a sibling executable. The isolated trusted PPO worker uses module invocation with user-site and environment-path loading disabled.

Historical documentation may quote former root paths as provenance. Those recorded command lines are not active callers; current launchers, tests, configs, and subprocesses use the grouped modules. Frozen evidence and admission records remain unchanged.

## Production pipeline

The supported application flow is:

| Stage | Implementation | Stable entrypoint |
| --- | --- | --- |
| Simulation | `src.core.simulation.Simulation` and `src.simulation` | Library API |
| Gymnasium | `src.rl.gym_environment.F1RLEnv` and `src.training.env_factory` | Library API |
| PPO training | `src.training.TrainingRunner`; registered campaign worker in `src.interfaces` | `python -m scripts.workflows training campaign` |
| Evaluation | `src.evaluation` and registered evaluation workers | `python -m scripts.workflows evaluation coordinate` |
| Inference | `src.interfaces.inference_cli` and `src.rl.algorithms.ppo.PPOModel` | `python -m scripts.workflows inference run-policy` |

Simulation and Gymnasium are library stages rather than standalone command-line programs. The WP3 campaign owns the registered training/evaluation sequence; its `training local-simulation` worker is retained because the campaign starts it as a subprocess. The removed demo handoff and one-click launchers do not participate in this pipeline.
