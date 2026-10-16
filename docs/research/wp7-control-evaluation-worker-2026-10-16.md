# WP7 explicit control and isolated PPO software worker

Parent: Draft PR #24, `c2768a3aa184b3862168eae092dc103f7567290d`.
This is engineering execution infrastructure, not admitted research training
or an empirical robustness result.

## Versioned control migration

`ActionContract()` retains `steering-throttle-brake0-v1` and two action values.
The explicit `steering-throttle-brake-v2` adds brake in [0,1] as a third value.
Pass it through `F1RLEnvFactory(action_contract=...)`; Gym and vector scenario
compatibility enforce the selected dimensions. No observation dimensions change.
Historical models keep two actions; loading one into a three-action environment
must fail the SB3 space check. No weight conversion is inferred.

Braking uses the existing registered ground longitudinal model. Tests check
deceleration and its non-reversal clamp through actual lifecycle Gym stepping.
In the registered gravity-only airborne domain steering/throttle/brake cannot
invent airborne forces. Observations remain finite and inside their declared
space through takeoff, flight and landing. This does not validate real tire slip,
braking distances on all courses, or a speed/boost schedule.

The experiment protocol remains on the historical two-action contract. Enabling
brake for a research experiment requires its own reviewed protocol/version.

## Execution identity

`wp7_policy_execution_binding_v2` extends the existing policy-bound receipt with
actual vector configuration, per-track race hashes, action version, observation
and simulation hashes, and checkpoint-selection plan. Validation independently
rebuilds the registered config and scenarios before artifacts or trainable model
creation. Changed vector settings or action contracts are rejected. V1 receipts
are stale; reconstruct with `build_policy_execution`, never silently migrate.
Run-time provenance and independently trusted approval remain separate gates.

## Real saved-model software inference

`execute_saved_ppo_fixture_cell` uses Windows-safe `spawn` and the existing cell
executor/evaluator. It accepts only exact legacy software fixture factories,
refusing research factories (including lying subclasses) before reading models.
It checks the expected archive SHA-256 and sends immutable verified bytes into
the worker, which loads the actual saved PPO model. Tests create an untrained
model (zero learned timesteps); no call to `learn` is made by the worker.

Each episode seeds Python/NumPy/PyTorch AFTER archive loading, since SB3 setup
can reset RNG state. Prediction is stochastic. Environment seeds use the separate
registered environment stream. Process isolation preserves parent RNG state.
The worker reports action count/hash and per-episode RNG start/end hashes, plus
ordinary metrics. Invalid episodes abort once without redraw and can be saved.
The canonical executed factory configuration, actual world hash, action contract
and observation features are recorded and reverified in the worker. The protocol
track label alone is not used to identify the synthetic environment.
Concrete component types and ordered reward names are included: dataclass field
values alone cannot distinguish different reward implementations.

The Torch-specific adapter lives in `scripts/wp7_ppo_fixture_worker.py` and is
started as a new Python process through a private trusted binary pipe. No Torch
import is added to `f1rl/`; existing dependency-boundary tests remain unchanged.
The pipe is internal IPC, not a decoder for external/untrusted evidence files.

The distinct `wp7_isolated_ppo_fixture_cell_v1` format records worker/device
identity and a per-cell exact one-sided CP95 lower bound. Minimum 60 trials and
lower bound >=.90 stay fixed; no pooling or familywise/cohort claim is inferred.
`scientific_evidence` and `scientific_acceptance` remain false even for a passing
software numerical cell. Cohort/multiplicity interpretation still needs review.

CPU and explicit CUDA are supported paths; unavailable CUDA is an error, never
fallback. This local host has CPU-only PyTorch, so CUDA execution is unverified.
Saved evidence is deterministic for the same exact source/model/runtime/cell;
cross-runtime bitwise reproducibility is not asserted.

`evaluate_registered_cell` still requires independent admission and remains
blocked. A functioning software worker does not authorize research rollouts,
HELD-OUT access, empirical acceptance, freeze or WP8.

Compatibility tightening: software fixture entrypoints now require the exact
`F1RLEnvFactory` type, preventing subclasses from hiding research admission flags.
External fixture subclasses must migrate to explicit factory instances.
