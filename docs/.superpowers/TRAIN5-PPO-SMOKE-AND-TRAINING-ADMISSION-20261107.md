# TRAIN-5 PPO frozen smoke and scientific training admission (2026-11-07)

Owner request: "PPO smoke và traing đi" (current conversation).

Target branch: `handoff/physics-v4-khanh-bang-20261106`.

## Separate frozen-policy smoke (approved scoped operation)

A **new, versioned** contract is stored in
`configs/research/wp7-train5-frozen-ppo-smoke-v1.json`. This is a
six-episode, source-bound, one-shot **inference** test. It does not
change the original TRAIN-5 v2 registration, the old four-track
FULL-06 protocol, frozen checkpoints, or any HELD-OUT file.

- Source base: `bb2d48bc17ba1e5bef75764998977fdc0cef27f0`.
- Models (in fixed order): `C-seed0`, `B-seed1`, `A-seed2`.
- TRAIN-only tracks (each): `right-turn-train-v2`, `left-turn-train-v2`.
- CPU `PPOModel.load`, deterministic predictions, reset seed `20261106`;
  episode upper bound 2,500 steps.
- One fixed output directory per committed runner SHA, located outside
  the git working tree. An attempt marker is created with exclusive
  creation before loading any policy. This prevents hidden reruns.
- Outcomes: finished, certified physical crash, valid hard-route failure,
  or valid step limit. Invalid Physics transitions, uncertified crashes,
  nonfinite actions/observations and runtime errors trigger fail-closed.
- Count all six valid case executions as smoke validity; *crashes are not
  policy success*. Keep terminal traces and action logs for diagnosis.

This new user-authorized runner does not call the old permanently denied
`require_smoke_admission` entrypoint; instead it performs an independent,
versioned user-authorized **smoke-only** admission for the exact six cases.
It never calls `PPO.learn`, any trainer, or HELD-OUT geometry, and does
not allocate one of the 18 training seed values.

## Scientific training remains a separate gate

`configs/research/wp7-train5-training-admission-review-v1.json`
records the user's request and the development budget proposed from the
already written 200k CPU/8-env pilot profile:

- Two cohorts (from-scratch and from frozen checkpoints), three seeds
  each, potential total 6 × 200k = 1.2M PPO timesteps;
- Independent reset/sampling/initialization seeds locked by existing
  TRAIN-5 proposal; no seed from that group has been consumed;
- Frozen finetune initial checkpoint candidates selected **before**
  smoke results, not selected to make smoke outcomes look better;
- Proposed 60 × 60 = 3,600 future TRAIN development episodes,
  **not** evaluations performed.

**Admission blockers are real:** the current draft always denies scientific
training, and the PPO backend and research trainer independently reject
learning on Physics v4 without an evidence-validated scientific admission
capability. Additionally, the new TRAIN-5 v4 reward/action/observation
equivalence, per-track reset distribution and empirical Physics validation
have not been admitted as a training research protocol. Merely asking to
train or closing the Global Seed Registry is not equivalent to satisfying
these scientific prerequisites.

The new protocol review is deliberately `BLOCKED`, not a permissive
switch. **Do not patch internal `requires_scientific_admission` flags,
forge owner approval, or use local exploratory demos as scientific training.**

## Safe execution sequence

1. Pre-register the exact six-case runner on clean source and record its
   independent smoke-only authorization.
2. Pass negative-path and training-denial tests and same-commit GitHub CI.
3. Execute the one-shot six-case smoke; report all outcomes without
   opportunistic reruns or retuning.
4. Re-review the training-only contract and actual smoke evidence.
   Training remains blocked unless the new verified scientific training
   admission path is separately accepted.
5. Any later training must save checkpoints outside the checkout and
   preserve the original 12 immutable model archives, and must not
   read HELD-OUT or run FULL-06.
