# WP3 A/B/C/D TRAIN Candidate Comparison

Evaluation status: **PASS_WITH_INVALID_CELLS**. Scope is simulation-only development; no scientific acceptance claim.

## Candidate finish rates

| Candidate | Seed 0 | Seed 1 | Seed 2 | Mean across seed blocks | Seed-block SD |
|---|---:|---:|---:|---:|---:|
| A | incomplete | incomplete | incomplete | incomplete | incomplete |
| B | incomplete | incomplete | incomplete | incomplete | incomplete |
| C | incomplete | incomplete | incomplete | incomplete | incomplete |
| D | 0.000 | incomplete | incomplete | incomplete | incomplete |

## Per-track finish-rate matrix

| Candidate | straight-sprint-v1 | slalom-v1 | right-turn-v1 | ramp-v1 |
|---|---:|---:|---:|---:|
| A | incomplete | incomplete | incomplete | incomplete |
| B | incomplete | incomplete | incomplete | incomplete |
| C | incomplete | incomplete | incomplete | incomplete |
| D | incomplete | incomplete | incomplete | incomplete |

## Stratum breakdown

| Candidate | Nominal | Randomized TRAIN |
|---|---:|---:|
| A | incomplete | incomplete |
| B | incomplete | incomplete |
| C | incomplete | incomplete |
| D | incomplete | incomplete |

## Secondary development outcomes

| Candidate | Mean route progress | Mean checkpoints | Collision rate | Hard-route violation rate | Timeout rate | Steering-rate RMS | Mean duration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.000 | 2.000 | 0.000 | 0.000 | 0.000 | 8.833 | 27.279 |
| B | 0.886 | 1.740 | 0.000 | 0.000 | 0.254 | 9.250 | 36.259 |
| C | 0.837 | 1.581 | 0.000 | 0.000 | 0.375 | 9.039 | 35.215 |
| D | 0.015 | 0.000 | 0.000 | 0.000 | 1.000 | 9.927 | 50.000 |

## Factor contrasts

| Contrast | Mean across PPO seeds | Seed-block SD | Per-seed differences |
|---|---:|---:|---|
| higher_lateral_penalty_minus_lower | incomplete | incomplete | incomplete |
| 3m_corridor_minus_2m_corridor | incomplete | incomplete | incomplete |
| factorial_interaction | incomplete | incomplete | incomplete |

## Failure-reason counts

- **A:** valid_finish=420
- **B:** step_limit=122, valid_finish=358
- **C:** step_limit=180, valid_finish=300
- **D:** step_limit=480

Factor contrasts are descriptive paired differences across the three seed blocks; they are not hypothesis tests.
Best descriptive candidate: **none (incomplete cells)**. Apparent differences are uncertain with three tuning seed blocks.

Charts: `WP3-MODEL-COMPARISON.svg`, `WP3-FAILURE-DISTRIBUTION.svg`, and `WP3-LEARNING-PROGRESS.svg`.

WP7 and Gate 2 remain **BLOCKED**.
