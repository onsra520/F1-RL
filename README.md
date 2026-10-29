# F1-RL

F1-RL is a Python-first autonomous racing and reinforcement-learning project. Its Gymnasium environment, deterministic track/vehicle simulation, PPO training/inference, research validation and the React/Three.js racing editor are separate parts of the repository.

## What works at the current engineering baseline

- Python application package is **`src/`** (not `f1rl/` or `src/f1rl/`).
- Independent wheel installation, model inference and frozen checkpoint/evidence integrity were verified at source `b11a501`.
- The frontend track editor builds and has 12 unit tests; it is **not yet connected** to a Python runtime service.
- Historical scientific records are verifiable as immutable artifacts, not regenerated as current-source approvals. Registered scientific evaluation and HELD-OUT remain **BLOCKED**.

**Not implemented:** the operational FastAPI/WebSocket UI↔Python bridge. It is an integration target, not an available server command. Avoid old README instructions suggesting `serve.py` exists.

## Layout

| Directory | Responsibility |
| --- | --- |
| `src/` | Python core, simulation, tracks, RL, training, evaluation, validation, interfaces |
| `web/` | React/TypeScript/Three.js racing sandbox |
| `scripts/` | Compatibility entrypoints and [versionless workflow router](scripts/README.md) |
| `configs/` | Versioned runtime/research configuration and registered profiles |
| `assets/` | Tracks and source assets |
| `tests/` | Python regression tests |
| `docs/` | [Current guides and historical research](docs/README.md) |
| `artifacts/` | Local development artifacts (not a source of scientific approval) |

## Workflows

From the repository root:

```powershell
python -m scripts.workflows list
python -m scripts.workflows list --json
python -m scripts.workflows evidence verify-frozen-assets
```

Commands that run training, simulation or research evaluation are **not automatically authorized** merely because they appear in the workflow registry. Keep the original approval and source-provenance requirements.

For setup, see [development docs](docs/development/README.md); for current limits and restore instructions, see [operations](docs/operations/README.md). The [architecture](docs/architecture/README.md) and [research](docs/research/README.md) indexes explain immutable boundaries.

## Safety and validation

Consolidation accepted 12 frozen PPO checkpoints and the 96-cell FULL-06 TRAIN-development evidence. Refactor validation reported 3,011 passing Python tests and 3 Windows symlink skips (retested on WSL), with the React frontend tests/build passing. These are **engineering** results; they are not independent scientific acceptance. No HELD-OUT access or registered evaluation is authorized by this repository navigation or workflow naming change.
