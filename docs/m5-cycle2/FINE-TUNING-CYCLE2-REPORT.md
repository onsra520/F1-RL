# M5 Improvement Cycle 2 — Fine-tuning Result

Date: 2026-11-11
Branch: `feat/m5-cycle2-c0-slalom-recovery-20261111`
Execution source commit: `e8f50e066a15ea848d08321ff1b59bf006fa63e0`
Execution source tree SHA-256: `a784bf414be3012f6df3a2da247788e2469ffa381fb27900401b91807c90dd60`

## Verdict

`FINE_TUNE_TARGET_NOT_MET`. The authorized single, bounded fine-tuning cycle started from immutable C0 std×0.5, trained for 4,096 additional PPO timesteps, evaluated its checkpoint on the pre-registered paired roster and stopped automatically after material Slalom regression. No checkpoint was eligible. The unchanged parent checkpoint remains the selected development baseline. M5 is not accepted; no model freeze or demo-ready claim follows.

## Admission and workload

- Parent checkpoint: C0 std×0.5, SHA-256 `b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86`.
- Parent optimizer SHA-256: `668fcdae8285d780c7033688d61fe9cb482fe7547e87ef14fa1370b994bd0c05`; initial policy-weight SHA-256 `dc2ca81e41b68ce2dd11afc287e767ea834188dcf8f3dfdef8eba6d44ce4550f`.
- Registered learned `log_std`: `[-0.9352163, -0.7101601, -0.8592330]`; optimizer step at admission: `10240`.
- Exact profile SHA-256: `5a1c3c6256b34662a3dc93851114c64f475397d943310caec7f5ebd5f8449b30`.
- Training/evaluation stayed CPU, one vector environment, five locked development tracks, existing Physics V4/action/observation/reward contracts, and 500 simulation-step episode limit. Right Turn V1, HELD-OUT, M2/M3 reruns, M6+, U-turn, Auto Drive and Hybrid were not used.
- Actual training: 4,096 additional steps; 16 PPO updates; 640 optimizer steps; finite gradients; 640 nonzero-gradient steps. Exposure: Left Turn V2 1,063; Ramp V1 901; Right Turn V2 838; Slalom V1 734; Straight 560. Total bounded attempt elapsed: 1,378.687 seconds including checkpoint evaluation.
- Checkpoint SHA-256: `8fdf18451a67da2d040f325b5c5c23cc6d8caca667f167bf4e89f625d43d7051`.

## Paired result

The paired roster was 85 cases: 15 stochastic episodes on each of five development tracks plus 10 deterministic Slalom diagnostics. Baseline and checkpoint used the same registered environment/policy seed pairs. The deterministic cell remains a diagnostic and is not a substitute for the stochastic contract.

| Track / mode | Parent std×0.5 | Checkpoint 69,632 | Selection interpretation |
|---|---:|---:|---|
| Straight stochastic | 15/15 | 15/15 | Preserved |
| Slalom stochastic, all | 12/15 | 11/15 | Regression; stop |
| Slalom stochastic, fixed | 8/10 | 7/10 | Material regression; stop |
| Slalom stochastic, fresh-development subset | 4/5 | 4/5 | Same |
| Right Turn V2 stochastic | 15/15 | 15/15 | Preserved |
| Ramp V1 stochastic | 15/15 | 15/15 | Preserved |
| Left Turn V2 stochastic | 15/15 | 15/15 | Preserved |
| Slalom deterministic diagnostic | 0/10 | 0/10 | No recovery; identical repeated trajectory within each cell |

Three parent Slalom stochastic failures occurred on seeds `427743197`, `2099305814`, `538351502`. The fine-tuned checkpoint failed on seeds `1010828269`, `2099305814`, `1346171939`, `1000129884`: it gained two formerly failing seeds and lost three formerly successful seeds, for a net loss of one finish. All four failures were physical road-edge crashes, not simulation step limits or wall timeouts. The full outcomes are in raw episode summaries and traces.

The checkpoint's aggregate stochastic Slalom mean normalized progress (`0.93430`) is higher than the parent's (`0.92905`) despite one fewer finish. Progress average does not override the registered finish gate.

## Integrity and stopping

- Evaluation report SHA-256: `222530e103cb2992084ee7e6e5171b656825bcc6c5b7061cb539a2a014155110`.
- Training result records one checkpoint, `continue_training=false`, `stopped_early`, and the exact parent retained as selected checkpoint. Selection report SHA-256: `ab5b83892e3198774ebebb346afba19d3e5db7525c17031f89467d8a229b17db`.
- Optimizer-state SHA after the one chunk: `1b3604e63342368b04d4ba264165a7fec42d60ec0964e07eb2e0b026cad3bc6d`; policy-weight SHA `62bb82a513ac8f5d1f3f3f8015ba0e7761c13a9d88b5fd6292a41100c4245f46`.
- All 85 report cases independently rehashed and reconciled against raw step traces. No held-out access, wall timeout, worker timeout or evaluation harness failure occurred.
- The prescribed fresh-roster candidate audit was **not run**: no eligible winner exists. Creating a post-selection audit roster without a selected checkpoint would not meet the pre-registered selection/audit separation.

## Required next step

Stop after Cycle 2. Preserve the original std×0.5 checkpoint as development baseline. Review the Slalom steering-policy failure and decide a new, evidence-supported change before authorizing any later improvement cycle. Do not extend this cycle or tune on the audit roster.
