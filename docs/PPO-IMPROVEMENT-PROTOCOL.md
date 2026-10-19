# Exploratory driving improvement, October 19, 2026

Original deliverable remains unchanged: model SHA-256
`b685284e35b39d094344aa9797f323c4def4666c808176ad89845d79994e660f`;
handoff archive SHA-256
`6fc7b97fa6b69f883d6fd52cb088926c69054fb59ebf526d42733b1c705a0a94`.

## Diagnosis before training

Evidence: `D:\Projects\.evidence\F1-RL\improvement\diagnostics-v1`.
Both original PPO failures reproduce at 29/31 steps, 14.54%/16.37% progress.
The model steers approximately -0.12 despite a positive target-local X direction,
with throttle approximately 0.95. Heading turns left to -0.39/-0.38 radians;
final yaw rates are -0.176/-0.180 radians/second. Lateral distance exceeds 4 m.
No collision or invalid transition causes these failures. Reward components
remain the registered route progress, collision and time terms, with no added
terminal adjustment. Small reset offsets/heading jitter match the intended domain.

The separately labeled scripted diagnostic controller uses target heading error
and yaw damping, and finishes both seeds in 145 steps with 100% progress.
This verifies route feasibility and the steering/observation sign convention;
it is not a learned policy and is never packaged as PPO.

These observations support undertrained steering feedback. They do not establish
a need for new reward semantics. Stage A retains registered rewards, actions,
observations, track identity and physics. Continuation is not implemented;
Stage A is a fresh PPO initialization, not a continuation of the old model.

## Performance

Fixed-input 80-step benchmarks at 4, 8, 16 and 32 threads produce the same
observation/reward/termination trace hash. Profiled wall times were 13.019,
12.819, 13.251 and 13.283 seconds, respectively. Physics/contact costs about
93% of profiled env.step time; sensors cost approximately 0.013 seconds across
80 steps, JSON formatting 0.08–0.12 seconds, checkpoint save 0.008–0.047 seconds.
These profiled times include profiler overhead.

Millions of Vec3 constructor generator validations were measured. Replacing
the generator with three direct finite checks preserves validation and yields
an identical trace. Unprofiled wall time fell from 3.458 to 2.695 seconds for
80 steps (23.14 to 29.69 steps/second, 1.28x in this bounded measurement).
No physical calculation or integration parameter changed. Eight CPU threads,
eight BLAS threads and one interop thread are configured in config.yml.
Rollout collection and optimizer time are recorded during authorized training;
no unapproved optimization/training workload is used for profiling.

## Prespecified staged evaluation

- Stage A: 8,192 fresh learning timesteps, checkpoints every 1,024 steps.
- PPO settings otherwise unchanged: n_steps 256, batch_size 64, n_epochs 10,
  learning_rate 0.0003, seed 0, CPU, one environment.
- Training reset seed 1000; tuning episodes 10000 and 10001.
- Stage B, if needed, is separately approved with a new exact permit.
- Reserved final validation seeds: 11000 through 11005. Do not use them for
  controller/hyperparameter tuning or intermediate model selection.
- Evaluate old and final candidate on identical seeds. Preserve every episode,
  including failure, collision, invalid samples and inference errors.
- Report success count, mean/median/worst progress, reasons, checkpoints and
  collisions for tuning and reserved seeds separately, and the combined roster.
- To replace the default, require valid inference and no errors on all reserved
  episodes, no reduction in success count, mean progress at least 0.05 above the
  original, higher median progress and no decrease in worst-case progress.
- Otherwise retain the original default model and handoff. Keep each archive
  and all run directories immutable; create a new package for a passing model.

Every learning run requires fresh direct owner approval bound to its committed
source, configuration, runtime, output path and exact one-use permit. Scientific
admission and HELD-OUT restrictions remain unchanged.
