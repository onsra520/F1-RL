# Spatial Ramp reset tracking correction

Parent: Draft PR #34, e1fac886ba194dd6fc9569ce4bab2f6b9460148f.
Branch: research/wp7-spatial-ramp-tracking-v1.

## Reproduced failure

The parent's four declared randomized cases pass, but its expanded fixed Ramp
cohort fails for every paired seed at rank 0/reset 0. Heading perturbations are
-2.4703499833, +3.7732989905 and -1.3932056390 degrees. The existing Ramp speed
controller always requests zero steering; lateral drift reaches the unchanged
hard corridor after 264, 195 and 416 accepted ticks. These invalid results are
preserved in the parent milestone evidence and are not redrawn as successes.

The public certificate regression reproduces seed 1's precise hard-corridor
failure in 13.44 seconds. The nominal Ramp completes and each failed reset
accepts many physical ticks, distinguishing missing route feedback from an
invalid initial state or an inherently infeasible track.

## Minimal correction

Only the explicit spatial TRAIN profile uses the new
`spatial_ramp_route_speed_stop_v1` controller. It reuses planar controller steering
against the authored horizontal navigation, retaining the existing slope-aware
Ramp speed/gravity/braking calculation. Existing heading/lateral gains are reused
without parameter tuning. Controller metadata records the new version and gains.
Legacy and standalone Ramp proposals keep their previous zero-steering behavior.
No physics configuration, owner threshold, canonical asset, RNG or seed changes.

All three originally failing selections now complete actual Simulation, ordered
center gates, body/road/contact/corridor checks, braking/stop and actual Gym with
valid observations. Seed 1 also matches exact source/runtime-bound replay.
Focused regression: **3 passed in 150.49 seconds** after the original single-case
RED/GREEN reproduction. Selected-profile software admission verification includes
this regression; training remains denied. Final full regression and compatibility
results are recorded separately after terminal completion.

This is finite scripted-controller engineering evidence, not learned PPO
performance, perturbation-range proof or statistical robustness. Spatial flight,
partial support, landing/recovery and physical calibration remain open. Readiness
weights/statuses remain unchanged and all four scientific permissions stay false.
