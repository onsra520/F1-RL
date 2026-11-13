# C0 Development Baseline Lock

Status: `C0_BASELINE_LOCKED_FOR_DEVELOPMENT`
Decision date: 2026-11-11
Scope: TRAIN/development only; this is not M5 acceptance evidence.

## Immutable model inputs

| Role | Artifact | SHA-256 | Provenance |
|---|---|---|---|
| Selected parent | `M4-C-seed0-std050-parent.zip` | `b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86` | M4 C-seed0 with only learned `log_std` scaled by 0.5; no gradient update |
| Comparator | `M4-C-seed0-original-comparator.zip` | `600e6eb3a781af66e34ea56f8f65e20f46d4a4318a77aeebceceacef3ffa7fd4` | Original 65,536-timestep M4 C-seed0 |

Both artifacts were copied byte-for-byte from the immutable archive. The recorded digests were verified before use. Neither input is modified in place. Each candidate checkpoint produced during Cycle 2 must be stored separately.

## Source and contracts

- Evaluation source anchor: `feat/m5-multicandidate-screening-20261111` at `112ca7b850ca811c6ae4ed58c54e80c8ba726bb2`; tree `f21d251c53367218f2dea0a827e11f175782edc9`.
- Physics: registered Physics V4 profile SHA-256 `c178b6a3109a7ae2d2c8b8b35f2e1e0868e0c97280ec5c83e643a7050d7c2e37`.
- Observation: 16 features, as defined by the five-track screening runtime factory.
- Action: `steering-throttle-brake-v2`; three continuous outputs, raw Pure PPO action passed through the registered mapper.
- Tracks: `straight-sprint-v1`, `slalom-v1`, `right-turn-train-v2`, `ramp-v1`, `left-turn-train-v2`.
- Episode limit: 500 simulation steps. Right Turn V1 is excluded from Cycle 2 runs.
- Reward: registered route-progress/collision/time reward from the screening factory; per-track runtime factory and reward hashes are recorded in the run plan before any training.

The fixed checkpoint-selection roster and the later fresh audit roster must be registered before use. Audit seeds remain unavailable to training, checkpoint selection, and early stopping. HELD-OUT remains sealed.

## Gate

The selected parent is a development baseline only. No M5 acceptance, scientific signoff, M6 freeze, or held-out claim follows from this lock. Fine-tuning may start only through a clean, committed, source-bound LOCAL_DEMO profile that validates this exact parent and the bounded five-track CPU scope. If that admission cannot be established without weakening the existing guard, training is blocked.
