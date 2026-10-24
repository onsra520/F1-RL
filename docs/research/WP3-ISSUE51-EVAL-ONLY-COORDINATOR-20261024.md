# WP3 Issue #51 - Post-lock Eval-only coordinator and acceptance boundary

Source branch: `fix/wp3-eval-physical-completion` (descendant of `ce0e55d`).
Original physics source-freeze tag: `wp3-issue51-physics-v4-crash-v1-locked-20261024`.
Training source `fe83f51` and 12 final PPO ZIPs remain immutable.

## Delivered

- `f1rl/research/wp3_eval_only_coordinator.py`: standalone, strict, no resume and no historical campaign writes.
- `scripts/wp3_eval_only_coordinator.py`: `preflight`, `smoke`, `full` CLI.
- `tests/research/test_wp3_eval_only_coordinator.py`: adversarial provenance, seed pairing, invalid episode, immutable model and idempotency tests.

The engineer-only smoke uses the canonical original paired seeds, final PPO checkpoint,
stochastic CPU inference and opt-in Physics v4 + Crash Terminal v1. Every attempt
writes a new distinct evidence root, a manifest SHA, per-cell append-only
`episode_started` then `episode_completed`/`episode_invalid` JSONL events,
per-run JSON, CSV and human-readable Markdown; no silent retries or overwrites.

The full 5,760-episode scientific Evaluation path is intentionally **BLOCKED**:
`preflight` checks original 12 final ZIP hashes, training telemetry/validation,
96-cell roster and source ancestry; `full` invokes the registered-admission
fail-closed gate and exits 3. `smoke` is deliberately capped at 32 development
episodes and may not be used to assert scientific candidate ranking.

## Commands

Run in a clean committed source worktree using the verified Python environment.

```powershell
python scripts/wp3_eval_only_coordinator.py preflight
python scripts/wp3_eval_only_coordinator.py full  # expected exit 3; no episodes
python scripts/wp3_eval_only_coordinator.py smoke --episode 4 --run-id WP3-ENG-ISSUE51-20261024-REPLAY2 --out D:\F1-RL-WP3-ISSUE51-EVAL-COORD-20261024
```

## Unresolved formal gate

The original Physics v4 technical receipt predates source lock and its exact
source-bundle hash differs from the locked successor in three integration files:
`f1rl/learning/gym_environment.py`, `f1rl/learning/wp7_race.py`,
`f1rl/learning/evaluation/evaluator.py`. These mismatches cannot be fixed by
rewriting historical evidence. A separately reviewed, source-bound v4 physical
acceptance, official evaluator/protocol registration, and independent signoff
are required before full registered TRAIN-only Evaluation; no automatic
acceptance, HELD-OUT, retraining or WP7 gate bypass is permitted.
