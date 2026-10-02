# `assets/tracks`

## Purpose

Portable, reproducible track definitions backed by the strict `f1rl.track_assets` loader and `manifest.json` semantic fingerprints.

## Candidate corpus

WP2 provides six primitive candidate tracks spanning road widths of 12–16 m, total authored surface length of roughly 120–160 m, 90-degree turns, lateral obstacle/waypoint variation, and a ramp rise of about 8.35 m. The canonical mechanics amendment adds one `boost-0` (+3.0 m/s²) to both slalom variants and one `slow-0` (×2.0 rolling-drag multiplier) to both turn variants. `straight-sprint-v1` and `ramp-v1` remain modifier-free control tracks. This corpus establishes both dimensional and mechanics evidence for the frozen WP1B environment.

## Subdirectories

- `training/` — candidate training-side organizational bucket.
- `evaluation/` — candidate evaluation-side organizational bucket.
- `demo/` — curated presentation/demo assets when added later.

The `training/` and `evaluation/` directory names remain organizational only. WP3 defines scientific membership in `configs/experiments/ppo-generalization-v1.yml`, and the resolver follows those declared IDs rather than inferring roles from paths. Selected research tracks must have globally unique `world_sha256` values so copied/renamed geometry or spawn-only variants cannot cross the split.

For protocol v1, pilot/tuning workflows use only the declared training IDs. Training therefore exposes obstacle, boost, slow-zone, and elevation mechanics before any held-out evaluation. The held-out pair reuses known obstacle/boost/slow mechanic classes while changing geometry; it does not introduce a novel dynamics mechanic. The declared held-out IDs remain reserved for final generalization evaluation.

Track JSON owns only world geometry/entities plus spawn pose. Experiment configuration and mutable runtime state must remain outside track files.
