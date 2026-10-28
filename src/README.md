# Python package

`src` contains the active Python implementation.

- `core`: simulation session orchestration and physics adapters.
- `simulation`: deterministic world, vehicle geometry and contact physics.
- `tracks`: portable track assets, validation, fingerprints and initial state.
- `rl`: actions, observations, rewards, Gym environment and PPO interface.
- `training`: training orchestration and configuration.
- `evaluation`: episode evaluation, metrics and crash contracts.
- `validation`: research gates, evidence verification and engineering attestation.
- `interfaces`: local demo and execution interfaces.
- `exploratory`: restricted exploratory execution.

Simulation must not import the track persistence or RL layers. Track persistence may use simulation domain types, but must not import RL, runtime frameworks or frontend code. Scientific identities and historical source paths remain recorded in immutable evidence.

An HTTP API, websocket transport and human driver adapter remain planned; no UI-to-Python bridge is implemented.
