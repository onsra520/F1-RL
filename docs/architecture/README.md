# Historical architecture snapshot — not the current system

This map is retained for reference only. It describes commit c0e1101 and predates Physics v4 and the Racing Sandbox merge. Its React/Three.js planned-only description is outdated: the browser editor exists but has no Python runtime bridge. Source lines and interfaces may also have moved. For current integration boundaries see ../research/WP3-UI-PHYSICS-V4-INTEGRATION.md.

# F1-RL architecture map

Open [F1-RL-architecture.html](F1-RL-architecture.html) in a browser for the interactive architecture diagram produced using [Archify](https://github.com/tt-a1i/archify).

The source specification is [F1-RL-architecture.archify.json](F1-RL-architecture.archify.json). It is backed by repository commit `c0e1101c2492acbe8c11f41a07485372f99d1950` (pinned in the spec).

The map covers PPO training, vector environments, Gymnasium, simulation and sandbox physics, track assets, reward/observation processing, and evaluation. FastAPI and React/Three.js are *planned adapters* documented by the project, not implemented services in this snapshot.

The diagram is a standalone HTML artifact. Archify source references in the JSON point to paths in this repository.

## Current baseline and refactor status

The current baseline map and dependency evidence are in [DEPENDENCY-MAP.md](DEPENDENCY-MAP.md); migration decisions and the currently blocked source-bound contract are in [MIGRATION-MAP.md](MIGRATION-MAP.md) and [FINAL-REFACTOR-AUDIT.md](FINAL-REFACTOR-AUDIT.md). This repository still uses `f1rl/`; the historical diagram above is not the current architecture.
