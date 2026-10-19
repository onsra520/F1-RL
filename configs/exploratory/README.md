# Exploratory PPO configuration v2

The repository root `config.yml` is the **only editable runtime source** for
the bounded local exploratory Straight pilot. No secrets or private keys belong
in it.

## Profiles and precedence

- `training`, `ppo`, `vector_env` are the existing general training profile.
  They remain compatible with WP6/WP7 recovery and scientific readers.
- `exploratory.*` scopes local Straight settings. It inherits the **root**
  `ppo` mapping, rather than redefining PPO parameters in multiple YAML files.
  It has its own explicit `training` and `vector_env` sections because its
  budgets differ from the general 50k / 8-env training profile.
- `configs/recovery/*` and `configs/research/*` are frozen/registered
  scientific inputs and are not mutable runtime overrides.
- No implicit environment-variable or CLI override wins over a resolved pilot
  permit. The CLI `--output-directory` remains the exact, explicit,
  **external-to-checkout** run destination; `artifacts_root` is descriptive
  training metadata rather than permission to relocate existing run evidence.

## Runtime behavior

`resources.cpu.threads: auto` and `blas_threads: auto` resolve using the
number of logical processors reported by the current runtime. Integer limits
must not exceed that number. This selects thread pool upper bounds; it cannot
magically parallelize a single Gym environment.

`exploratory.training.total_timesteps` and
`checkpoint_every_steps` must align with `ppo.n_steps`. The exploratory
profile remains CPU-only, one dummy environment, registered TRAIN Straight
physics, no HELD-OUT, with a maximum 32,768 interactions, 3,600-second wall
clock, and 32 evaluation episodes per policy. Those ceilings are **guardrails
in code**, not arbitrary tunables that confer research permission.

`execution.schedule` currently accepts only `baseline_first`: although a
`learning_first` schedule is desirable, it requires its own tested
orchestration change and fresh owner approval. A YAML switch must not claim to
activate unimplemented scheduling.

## Preflight and provenance

A strict loader rejects duplicate/unknown keys, wrong types, unsupported
execution modes, excessive budgets and inconsistent rollout/batch settings.
It resolves all runtime values **before** preparing the exploratory permit.
The raw `config.yml` SHA-256, resolved budgets/seeds/hyperparameters, runtime
identity and clean repository source are bound into a new one-use permit.

Changing configuration, checkout bytes or runtime after preflight invalidates
the permit; obtain a new preflight and a new local owner consent. Never reuse an
old output directory, restart a consumed authorization, mutate old evidence,
enable WP7 scientific admission or unseal HELD-OUT.

The local approval acknowledgement is now
`APPROVE_ONE_LOCAL_EXPLORATORY_RUN` and is tied to the exact permit SHA-256,
with expiry also bounded by the configured wall-clock budget. Old approval
envelopes intentionally fail closed.

## Verification

Run locally on the **checked-out ref** (not another installed F1-RL package):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -B -m pytest -q tests/research/test_exploratory_config.py tests/research/test_exploratory_permit.py tests/research/test_exploratory_pilot.py tests/learning/training/test_run_config.py
```

These are tests only, not a grant to execute `PPO.learn()`.
