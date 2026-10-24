# WP3 Issue #51 | New engineering Eval-only evidence

Run ID: `WP3-ENG-ISSUE51-20261024-REPLAY2`
Source HEAD: `0886a68dd1a8e17c0592f2399380665fef3d6734`
TRAIN SHA: `fe83f517479f3b990028d6409304fae83b8199a9`
Roster SHA-256: `f2f9d15954cea48345a8932773ed945f6ed250ad1435b4e675b97e4a6b94f319`

**UNREGISTERED ENGINEERING ONLY — full 5,760-slot scientific Eval BLOCKED.**
No training, no HELD-OUT, no modification of historical Eval.

Completed 16/16; invalid 0.

| Model | Completed | Finish | Physical crash | Step limit | Invalid |
|---|---:|---:|---:|---:|---:|
| A-seed1 | 8 | 4 | 3 | 0 | 0 |
| C-seed0 | 8 | 4 | 2 | 2 | 0 |

## Per-episode results

| Model | Track | Stratum | Episode | Outcome | Progress* |
|---|---|---|---:|---|---:|
| A-seed1 | straight-sprint-v1 | nominal | 4 | finish | 1.0 |
| C-seed0 | straight-sprint-v1 | nominal | 4 | finish | 1.0 |
| A-seed1 | straight-sprint-v1 | randomized_train | 4 | finish | 1.0 |
| C-seed0 | straight-sprint-v1 | randomized_train | 4 | finish | 1.0 |
| A-seed1 | slalom-v1 | nominal | 4 | physical_crash | 0.9167469144280477 |
| C-seed0 | slalom-v1 | nominal | 4 | finish | 1.0 |
| A-seed1 | slalom-v1 | randomized_train | 4 | physical_crash | 0.9119529440039944 |
| C-seed0 | slalom-v1 | randomized_train | 4 | finish | 1.0 |
| A-seed1 | right-turn-v1 | nominal | 4 | physical_crash | 0.5071219644544244 |
| C-seed0 | right-turn-v1 | nominal | 4 | physical_crash | 0.503422106016523 |
| A-seed1 | right-turn-v1 | randomized_train | 4 | hard_route_violation | 0.5328611350206305 |
| C-seed0 | right-turn-v1 | randomized_train | 4 | physical_crash | 0.5044972120312401 |
| A-seed1 | ramp-v1 | nominal | 4 | finish | 1.0 |
| C-seed0 | ramp-v1 | nominal | 4 | step_limit | 0.8391839733362525 |
| A-seed1 | ramp-v1 | randomized_train | 4 | finish | 1.0 |
| C-seed0 | ramp-v1 | randomized_train | 4 | step_limit | 0.7630562120498786 |

*Uncertified engineering telemetry, not scientific performance.
Only a small number of seeds were sampled; no candidate rankings allowed.
New independently reviewed physics certification and a separately registered source/protocol are mandatory before full Eval.
