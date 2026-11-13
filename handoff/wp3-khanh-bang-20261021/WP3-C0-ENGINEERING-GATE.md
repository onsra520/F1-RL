# WP3 C0 Full-Length PPO Engineering Gate

Status: **ENGINEERING_PASS**

- Model: `C-seed0` (candidate C, seed index 0)
- Source commit: `fe83f517479f3b990028d6409304fae83b8199a9`
- Learning timesteps: 65,536
- PPO update rows: 256
- Optimizer steps: 10,240
- Initial policy weights SHA-256: `02f3273342ec475ea457ed01af9104e13c5c565dce7ed3864eb5f378d5155e39`
- Final policy weights SHA-256: `365c404762f612e245f6922576a15ca40f5e3d858631808f8c16e5fb8ac0a389`
- Telemetry SHA-256: `9d80d3262ec2d134a0ebb5c733abf7d0965da97d6027b93638037409928096f5` (269,773,075 bytes)
- Checkpoints verified: 17 archives (16 periodic plus final)
- Episodes: 254; automatic reset boundaries verified: 253
- Natural terminations: 44; curriculum terminations: 112; truncations: 28
- Unsupported-contact training terminals: 69
- Runtime split: TRAIN only; HELD-OUT access and scientific evaluation: disabled.

## Observed registered TRAIN coverage

| Track | Timesteps | Episodes |
|---|---:|---:|
| straight-sprint-v1 | 12,870 | 48 |
| slalom-v1 | 16,319 | 72 |
| right-turn-v1 | 13,846 | 68 |
| ramp-v1 | 22,501 | 66 |

Driving success is not part of this engineering gate.

WP7 and Gate 2 remain **BLOCKED**.
