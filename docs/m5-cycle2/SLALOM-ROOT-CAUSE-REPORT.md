# Slalom Deterministic Recovery — Root Cause

Date: 2026-11-11
Execution source: `e8f50e066a15ea848d08321ff1b59bf006fa63e0`
Source tree SHA-256: `a784bf414be3012f6df3a2da247788e2469ffa381fb27900401b91807c90dd60`
Scope: TRAIN/development; Pure PPO, Physics V4; no HELD-OUT access.

## Finding

The registered deterministic Slalom failure is a policy control failure at the rightward transition. The deterministic policy mean continues full right steering while the vehicle is already moving right, then switches to full left only after it has overshot. The car crosses the road edge and terminates in a physical road-body crash. The same trajectory is reproduced across all 10 nominal deterministic seed pairs, so those seed labels do not represent 10 distinct physical trajectories in this evaluation factory and cannot establish robustness.

The evidence does not demonstrate a defect in Physics V4, action mapping, reward, route geometry, checkpoint ordering, or terminal handling. It also does not isolate vehicle speed as a cause. The direct failure mechanism is late corrective steering followed by edge contact; speed may affect recoverability, but that contribution remains unproven.

## Paired observations

| Measure | Parent `std×0.5` | Fine-tuned checkpoint 69,632 |
|---|---:|---:|
| Deterministic Slalom | 0/10 | 0/10 |
| Identical deterministic trajectory | Yes, 10/10 | Yes, 10/10 |
| Deterministic terminal step | 98 | 90 |
| Final normalized progress | 0.4414936 | 0.4247125 |
| Final position `(x,y,z)` | `(5.8423, 0.4634, 57.8647)` | `(6.0913, 0.4270, 55.7471)` |
| Terminal | `physical_crash`, road-main edge | `physical_crash`, road-main edge |
| Deterministic mean speed | 6.020 m/s | 6.407 m/s |

On the parent diagnostic trajectory, at step 80 the vehicle is near `x=0.02m, z=52.18m`, with about `+1.74m` signed lateral route error and `+0.31rad` yaw error while commanding full right steering and full throttle. Around step 90 it is at `x=4.83m, z=55.18m`, yaw error about `+0.82rad`; full-left correction begins after the overshoot. It reaches the road edge at step 98. The fine-tuned trajectory terminates earlier at step 90 with the same road-main contact class.

The Cycle 2 checkpoint evaluation also measured Slalom stochastic applied-control averages weighted by steps:

| Measure | Parent | Fine-tuned |
|---|---:|---:|
| Mean steering | −0.0324 | −0.0401 |
| Mean throttle | 0.6371 | 0.6488 |
| Mean brake | 0.0175 | 0.0253 |
| Mean speed | 4.889 m/s | 4.879 m/s |

These averages do not isolate a cause; they show that the fine-tuning attempt did not produce deterministic recovery and did not improve the paired stochastic finish count.

## Confirmed vs. unresolved

- **Confirmed:** deterministic policy-mean control fails at the Slalom rightward transition and ends in a physical road-edge collision.
- **Confirmed:** deterministic seed labels repeat an identical trajectory, so deterministic 10/10 is not ten independent outcomes.
- **Confirmed:** action observations distinguish raw PPO output, action delivered to the environment, and mapped controls. No mapping mismatch was found in these runs.
- **Not established:** excessive speed as a standalone cause; a Physics V4 defect; reward/curriculum defect; a single policy parameter that will restore deterministic finish.
- **Outcome:** Cycle 2's single 4,096-step chunk did not recover deterministic Slalom. It reduced the fixed stochastic Slalom roster from 8/10 to 7/10, triggered the predeclared material-regression stop, and was not selected.

## Evidence

- Parent report: `D:\Projects\.archives\F1-RL\demo-completion\m5-cycle2-c0-slalom-recovery-20261111\training-attempt-v5\parent-selection-baseline\EVALUATION-REPORT.json`, file SHA-256 `6206d4c8bd55921a4b5cc2eb72eaada05e320fa9eee53081fd8106aa2498f210`.
- Fine-tuned report: `D:\Projects\.archives\F1-RL\demo-completion\m5-cycle2-c0-slalom-recovery-20261111\training-attempt-v5\checkpoint-evaluations\step-69632\EVALUATION-REPORT.json`, file SHA-256 `222530e103cb2992084ee7e6e5171b656825bcc6c5b7061cb539a2a014155110`.
- Fine-tuned checkpoint SHA-256 `8fdf18451a67da2d040f325b5c5c23cc6d8caca667f167bf4e89f625d43d7051`.
- Independent raw-trace audit: 85/85 report cases, trace hashes, seed pairs, action/speed aggregates, terminal consistency and stochastic track groups verified. See `INDEPENDENT-AUDIT-REPORT.md` and archived `training-attempt-v5\INDEPENDENT-AUDIT.json`.
