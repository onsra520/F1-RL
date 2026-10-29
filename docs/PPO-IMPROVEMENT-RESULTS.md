# Historical / Retired Legacy — PPO driving improvement (October 19, 2026)

**Historical / Retired Legacy.** This report preserves results and launch-path
provenance from the October 2026 demo work. `START-DEMO.cmd` is no longer
supported and the file has been removed; use the current workflow registry for
supported inference and campaign entrypoints.

Historical run commands below record the launch paths used at the time. The
`START-DEMO*.cmd` launchers and lecturer handoff packager were retired on
2026-10-29; current supported workflows are listed in
[`docs/development/ENTRYPOINT-ARCHITECTURE.md`](development/ENTRYPOINT-ARCHITECTURE.md).

**VERIFIED: the new genuine PPO drives farther on the original task.**
**PARTIAL: it still finishes 0/8 evaluation episodes; all end in hard-route failure.**

## Diagnosis and changes

The original policy steers left when the target is right. On seed 10000 its
initial action is [-0.12681, 0.96250, 0], with final heading -0.39284 radians
and yaw rate -0.17575 radians/s. This grows into a 4 m route exit. The new
policy starts at [-0.03071, 0.83040, 0]; its final heading is -0.11674 radians
and yaw rate -0.04644 radians/s. Steering bias and drift are reduced, but
corrective feedback is still insufficient. These measurements support a
policy-quality problem, not a sign reversal in the actuator mapping.

The separately identified scripted diagnostic controller finishes both tuning
seeds using only the original observation. It is never substituted for PPO,
counted as a learned result, or included as the submitted model.

Stage A failed on an invalid body/contact transition after 4,243 accepted
learning steps / 4,244 attempted steps and 640 optimizer steps. Constant-input
diagnostics reproduce the same error class. The physics guard is preserved;
no invalid sample is relabeled as a normal failure. Stage A evidence remains
at `D:\Projects\.evidence\F1-RL\improvement\stage-a-8192`.

Approved Stage B uses a different, explicit **demo-only training task**:
`straight_route_stability_demo_v1`. After each valid physics step, a route
distance over 2 m ends the training episode. Training reward is registered
reward minus `2 * distance_m^2 * dt`, minus 25 on a curriculum stop or registered
hard-route failure. Original research outcomes and all penalty components are
recorded separately. Evaluation/inference use the original 4 m task and
registered rewards. Training returns are therefore not comparable across tasks.

Production changes are limited to this scoped curriculum, permit/config binding,
timing evidence, inference/paired evaluation, packaging and a physics-equivalent
Vec3 validation optimization. The latter preserves exact trace hashes and
improved a bounded unprofiled workload from 23.14 to 29.69 steps/s (1.28x).
The final verification suite passed **808 tests**.

## Training evidence

- Branch: `demo/ppo-driving-improvement-20261019`.
- Training commit: `3c3d7d3823703f1edb140029bd6989d5d359cc80`.
- Owner-approved permit: `e84976cc5fedfbce28a3fa15d3409037d80c641635584318bbcf2e712d9aa30d`.
- Fresh PPO initialization; neither Stage A nor Stage B continues the original model.
- CPU only; one environment, eight Torch/BLAS threads, one interop thread.
- 16,384 timesteps; n_steps 256, batch_size 64, n_epochs 10, learning_rate 0.0003.
- PPO seed 0, reset seed 1000; checkpoint every 1,024 steps.
- 2,560 optimizer steps; 640 PPO epoch updates; weights genuinely changed.
- PPO.learn: 1,177.318 s; rollout collection: 1,169.785 s; optimizer: 7.108 s.
- Full supervised run including baselines and replay: 1,304.890 s, within the 3,600 s cap.
- Final model SHA-256: `b3524af0eaeff3458293e2f59a271b590cb9a57a0c0f626569bd9cd1e6d3505f`.
- Final weight-state SHA-256: `2f6fc6a837485019b1796690612d40a062a3ab49d116cdd90eadcb6ca1c9db75`.
- Evidence: `D:\Projects\.evidence\F1-RL\improvement\stage-b-stability-16384\run`.

## Prespecified paired original-task results

Seeds: 10000, 10001 (tuning), and 11000–11005 (six reserved from tuning).
No episode was discarded. All 16 old/new episodes are valid, with no inference
errors or collisions. Every episode ends in a hard-route violation.

| Roster / metric | Original PPO | New PPO |
|---|---:|---:|
| All eight: mean progress | 17.31% | 38.71% |
| All eight: median progress | 17.49% | 39.86% |
| All eight: worst progress | 13.81% | 27.54% |
| All eight: successful finishes | 0/8 | 0/8 |
| All eight: checkpoint passes | 0 | 7 |
| Reserved six: mean progress | 17.93% | 40.28% |
| Reserved six: median progress | 18.18% | 42.42% |
| Reserved six: worst progress | 13.81% | 27.54% |

The candidate passes the prespecified replacement criterion: valid inference,
no reduced success count, mean progress at least 5 percentage points higher,
higher median and no decrease in minimum progress on reserved seeds.
This is measured driving improvement, not a claim of successful autonomous
driving, generalization, or scientific certification.

Paired evidence:
`D:\Projects\.evidence\F1-RL\improvement\stage-b-final-paired-evaluation`.
Step telemetry for the new model:
`D:\Projects\.evidence\F1-RL\improvement\stage-b-telemetry`.

## Demo and new handoff

From `D:\Projects\F1-RL`:

```text
START-DEMO.cmd
```

The verified local default is the separately named
`models/ppo-straight-improved-16384.zip`. To demonstrate the preserved original:

```text
START-DEMO.cmd --model models\ppo-straight-trained.zip
```

Both Windows launch paths were executed successfully without training. The
model files remain local generated artifacts, excluded from Git.

New archive:
`D:\Projects\.evidence\F1-RL\improvement\handoff-stage-b-16384\F1-RL-DEMO-HANDOFF.zip`.

Archive SHA-256:
`183a85cb49b1a5e7b52d605c8a46b9634c51c4e774958d00eb66870e24ad2b7b`.

It includes the genuine final model, task/configuration provenance, pinned
dependencies, contracts, all paired outcomes, source, required TRAIN assets,
README and checksums. A fresh extraction passed all 158 file hashes and
reproduced all eight paired traces. An audit hook denied access to the original
repository and original model/run directories during inference.

The existing dependency interpreter was reused. Fresh dependency installation
and other operating systems were not tested. The CLI reports real progress and
failure outcomes; no new GUI or successful finish is claimed.

The original model SHA-256 remains
`b685284e35b39d094344aa9797f323c4def4666c808176ad89845d79994e660f`.
The previous handoff archive SHA-256 remains
`6fc7b97fa6b69f883d6fd52cb088926c69054fb59ebf526d42733b1c705a0a94`.
Both remain intact. HELD-OUT remains unused, scientific admission and frozen
contracts remain unchanged, and all failed/prior run evidence is retained.
