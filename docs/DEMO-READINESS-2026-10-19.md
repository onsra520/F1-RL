# F1-RL demo readiness — October 19, 2026

**Historical record.** The root launchers described in the original run record were retired on 2026-10-29. This document retains the run results and command names as provenance; use the current workflow registry for supported entrypoints.

The genuine exploratory PPO model trains, reloads, and runs inference. It does
**not** successfully complete the demonstration track: both evaluated episodes
ended in hard-route failures. Present it as a working RL training/inference
pipeline with limited policy quality.

## Current pipeline entrypoints

The supported training/campaign and saved-policy inference entrypoints are:

```text
python -m scripts.workflows training campaign --help
python -m scripts.workflows inference run-policy --help
```

The old commands `START-DEMO.cmd` and `START-DEMO-TRAIN.cmd` are retired. The
historical training results and launcher behavior below remain a record of the
October 2026 demo; they do not describe supported launch instructions.

## Actual authorized run

- Training commit: `e07ceaa9f3c1503a61c1cb57cb49718ae3983d90`.
- Run: `lecturer-smoke-isolated-20261019`.
- Evidence: `D:\Projects\.evidence\F1-RL\demo\lecturer-smoke-isolated-20261019\run`.
- Permit: `6dd51ec045da6c97a118228b6173d6a69810c212283fa8c839b41ec1444b7c56`.
- Config SHA-256: `bc64175b308a547fd053320897279dfe84ff5e3b9d11328d02130238476004ae`.
- CPU only, one environment, 32 Torch/BLAS threads, one interop thread.
- Schedule: `learning_first`; 4,096 learning timesteps, 640 optimizer steps,
  160 PPO epoch updates. Complete run: 462.547 seconds; cap: 3,600 seconds.
- Training seed: PPO 0; reset 1000. Evaluation seeds: 10000 and 10001.
- Initial weights: `02f3273342ec475ea457ed01af9104e13c5c565dce7ed3864eb5f378d5155e39`.
- Final weights: `cb59f5d33fcc25ac76c50468f65f7d79c469bd7a5406d54374c465eccffd3216`.
- Final model SHA-256: `b685284e35b39d094344aa9797f323c4def4666c808176ad89845d79994e660f`.
- Checkpoints retained every 512 steps, plus immutable untrained and final models.

## Results

All six evaluation episodes are valid samples. None finished successfully.
No collisions occurred. Each trained episode recorded one off-route terminal event.

| Policy | Progress, seed 10000 / 10001 | Return, seed 10000 / 10001 | Outcome |
|---|---|---|---|
| Trained final | 14.54% / 16.37% | 15.419 / 17.382 | Hard-route failure at 2.9 / 3.1 simulation seconds |
| Random | 6.73% / 5.60% | 2.270 / 1.047 | 500-step truncation at 50 simulation seconds |
| Original untrained | 0.58% / 0.62% | -4.369 / -4.333 | 500-step truncation at 50 simulation seconds |

No checkpoints were completed. No inference errors occurred. Deterministic
replay matched exactly. Separate saved-model inference reproduced both trained
episode traces. These two seeds do not support generalization claims.

## Lecturer handoff

Final directory:
`D:\Projects\.evidence\F1-RL\submission\verified\F1-RL-DEMO-HANDOFF`.

Final archive: the same path with `.zip` appended.

It includes the real model, runtime source, required TRAIN assets, original
configuration, pinned dependency versions, observation/action contracts, all six
evaluation outcomes, training provenance, README, and SHA-256 checksums.
Fresh-extraction verification and its results are retained beside the archive.
Verification uses the installed dependency interpreter; dependency installation
on a new machine and other operating systems are not claimed as tested.

## Verification and boundaries

The focused suite passed with 443 tests and one skip after the inference
launcher fix. Package creation additionally reloads the final model and checks
its timesteps and weight hash against the genuine learning summary.

The initial candidate's preflight statically opened HELD-OUT assets while
validating the manifest. This was found before training and corrected in the
TRAIN preparation path. A regression denies HELD-OUT asset opens during scenario
construction. No HELD-OUT rollout or result evaluation was performed.

Scientific admission remains unchanged. Historical evidence, physics, rewards,
observation/action semantics, track identities, and research registration were
preserved. All run evidence and prior package candidates remain intact.
