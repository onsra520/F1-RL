# `assets/tracks`

## Purpose

Portable, reproducible track definitions backed by the strict `f1rl.track_assets` loader and `manifest.json` semantic fingerprints.

## Candidate corpus

WP2 provides six primitive candidate tracks spanning road widths of 12–16 m, total authored surface length of roughly 120–160 m, 90-degree turns, lateral obstacle/waypoint variation, and a ramp rise of about 8.35 m. This geometry establishes dimensional evidence for the later WP1B environment/sensor freeze.

## Subdirectories

- `training/` — candidate training-side organizational bucket.
- `evaluation/` — candidate evaluation-side organizational bucket.
- `demo/` — curated presentation/demo assets when added later.

The `training/` and `evaluation/` directory names remain organizational only. WP3 defines scientific membership in `configs/experiments/ppo-generalization-v1.yml`, and the resolver follows those declared IDs rather than inferring roles from paths. Selected research tracks must have globally unique `world_sha256` values so copied/renamed geometry or spawn-only variants cannot cross the split.

For protocol v1, pilot/tuning workflows use only the declared training IDs. The declared held-out IDs are reserved for final generalization evaluation.

Track JSON owns only world geometry/entities plus spawn pose. Experiment configuration and mutable runtime state must remain outside track files.
