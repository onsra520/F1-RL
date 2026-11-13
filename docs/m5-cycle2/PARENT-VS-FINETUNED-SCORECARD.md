# Parent vs. Fine-tuned Checkpoint Scorecard

Source commit: `e8f50e066a15ea848d08321ff1b59bf006fa63e0`
Evaluation: paired 85-case roster; Pure PPO on TRAIN/development; Physics V4.

| Track | Mode | Parent finishes | Checkpoint 69,632 finishes | Progress parent → checkpoint | Parent crashes → checkpoint |
|---|---|---:|---:|---:|---:|
| Straight | stochastic | 15/15 | 15/15 | 1.000 → 1.000 | 0 → 0 |
| Slalom | stochastic | 12/15 | 11/15 | 0.9291 → 0.9343 | 3 → 4 |
| Right Turn Train V2 | stochastic | 15/15 | 15/15 | 1.000 → 1.000 | 0 → 0 |
| Ramp | stochastic | 15/15 | 15/15 | 1.000 → 1.000 | 0 → 0 |
| Left Turn Train V2 | stochastic | 15/15 | 15/15 | 1.000 → 1.000 | 0 → 0 |
| Slalom | deterministic diagnostic | 0/10 | 0/10 | 0.4415 → 0.4247 | 10 → 10 |

Slalom deterministic seed-pair cases all repeat one trajectory per checkpoint, so the count is a reproducibility diagnostic and does not represent ten independent trajectory outcomes. Both deterministic cells crashed at the rightward transition. The stochastic fixed-roster cell dropped from 8/10 to 7/10; the fresh-development subset remained 4/5. The predeclared gate stopped further fine-tuning.

Checkpoint file SHA-256: `8fdf18451a67da2d040f325b5c5c23cc6d8caca667f167bf4e89f625d43d7051`.
Parent checkpoint SHA-256: `b6de69382523b9dc5309a6cffb230afd31f9b0206d3381dff723abc69ebffa86`.
Parent evaluation report SHA-256: `6206d4c8bd55921a4b5cc2eb72eaada05e320fa9eee53081fd8106aa2498f210`.
Checkpoint evaluation report SHA-256: `222530e103cb2992084ee7e6e5171b656825bcc6c5b7061cb539a2a014155110`.
