# Experiment configurations

This directory contains canonical scientific experiment definitions. These files are not generic runtime presets: they record the protocol whose results will later be compared and reproduced.

## `ppo-generalization-v1.yml`

`ppo-generalization-v1.yml` is intentionally checked in as a **draft** during WP1A. Research-critical environment numbers remain explicit `null` values, the rangefinder suite is empty, and seed arrays are empty. No value from tests or benchmarks is promoted into the research protocol merely because it already exists in the repository.

The draft references root `config.yml` as the current development/runtime training configuration and `assets/tracks/manifest.json` as the future WP2 track manifest. A draft may reference the not-yet-created manifest; it is not runnable as a frozen experiment.

WP2 defines authored track data and representative geometry scale. WP1B then resolves vehicle, simulation, observation/sensor, reward, and episode values. WP7 pilots training-only choices and freezes the final training profile. Only after all research-critical inputs are resolved may the definition change to `status: frozen` and pass `ExperimentDefinition.require_frozen()`.

Once frozen, materially changing physics, sensors, reward, episode horizon, final training protocol, track split, or evaluation semantics requires a new protocol version rather than silently editing v1 after held-out results are observed.
