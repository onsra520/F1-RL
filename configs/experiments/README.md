# Experiment configurations

This directory contains canonical scientific experiment definitions. These files are not generic runtime presets: they record the protocol whose results will later be compared and reproduced.

## `ppo-generalization-v1.yml`

`ppo-generalization-v1.yml` remains an explicit **draft** experiment, but WP1B has now pinned its environment protocol. Vehicle dynamics, simulation timestep/surface-query distance, the seven-rangefinder declaration, observation scales, reward weights, and episode horizon are resolved to the reviewed protocol-v1 values.

The experiment is intentionally **not** whole-protocol frozen yet. Root `config.yml` remains the current development/runtime training reference, `training.protocol_status` remains `pilot_owned`, and the PPO/sampling seed arrays remain empty. WP7 owns pilot calibration and the final training profile; the later scientific freeze also requires an explicit replicate-seed protocol before `ExperimentDefinition.require_frozen()` may pass.

The track reference points at the concrete WP2 manifest in `assets/tracks/manifest.json`. The current `training/` and `evaluation/` asset directory names are organizational only; WP3 owns the scientific train/held-out split and leakage guards.

The WP1B environment values are protocol material. Changing vehicle/simulation values, sensor count/order/directions/ranges, observation scales, reward weights, or episode horizon requires explicit protocol review and, once the baseline is frozen or used for held-out results, a new protocol version rather than silently editing v1.

No value is inherited from tests merely because a similar fixture exists: the WP1B values are justified by the implemented dynamics and the authored WP2 corpus scale.
