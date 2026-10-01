# `assets/tracks`

## Purpose

Portable, reproducible track definitions backed by the strict `f1rl.track_assets` loader and `manifest.json` semantic fingerprints.

## Candidate corpus

WP2 provides six primitive candidate tracks spanning road widths of 12–16 m, total authored surface length of roughly 120–160 m, 90-degree turns, lateral obstacle/waypoint variation, and a ramp rise of about 8.35 m. This geometry establishes dimensional evidence for the later WP1B environment/sensor freeze.

## Subdirectories

- `training/` — candidate training-side organizational bucket.
- `evaluation/` — candidate evaluation-side organizational bucket.
- `demo/` — curated presentation/demo assets when added later.

The `training/` and `evaluation/` names are organizational only in WP2. WP3 owns the scientific train/held-out split and leakage rules.

Track JSON owns only world geometry/entities plus spawn pose. Experiment configuration and mutable runtime state must remain outside track files.
