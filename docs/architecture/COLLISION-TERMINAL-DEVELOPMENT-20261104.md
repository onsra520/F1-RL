# PR #57 — Runtime-only collision failure (development)

## Scope (fixed)

- Custom Auto Drive with Physics v3, cone/barrier obstacles.
- No change to TRAIN/PPO, frozen models, HELD-OUT, Physics v4 crash impulse contracts or scientific acceptance.
- The user-facing result is **collision -> runtime diagnostic penalty -> failed episode -> reset**.
- The penalty is a development diagnostic, **not** a Gym reward; collision data is **not** admissible RL evidence.

## Source-of-truth boundary

1. `step_world` retains its fail-closed `FlightPhysicsError` when it witnesses a body hit but cannot accept a Physics collision transition. The witnessed obstacle IDs carry no time ordering.
2. The Gym environment marks the physical transition invalid and raises. No Gym return, reward or accepted Physics step is invented.
3. The **optional runtime-only** `evaluate_policy_episode(on_error=...)` callback inspects that exact witnessed v3 exception and records an out-of-band collision terminal. Other errors, uncertain sweeps, PPO and v4 remain unchanged.
4. The last frame remains the last accepted Physics snapshot. Collision IDs, unscored penalty and witness tick interval appear **only** in `collision_terminal` and the out-of-band WebSocket collision event, not as a fabricated replay frame.
5. Runtime result: `status=failed`, `reason=collision`, `valid_sample=false`, `scientific_evidence=false`. No fake impulse, first-contact timestamp, checkpoint or route reward.
6. Reset from the recognized collision failure starts a fresh episode epoch on the same session. Previous replay/terminal is not appended to the new episode.

## Acceptance gates

- Real cone + barrier witness: terminal metadata, accurate last accepted tick, monotonically ordered accepted frames, no fabricated collision frame and no false valid sample.
- Diagnostic penalty equals the configured `CollisionPenalty` (-5 for current custom factory), applied once **outside** the Gym reward/rollout return.
- Post-failure reset starts a distinct deterministic episode; state/replay are regenerated with tick 0.
- Unknown IDs and uncertain contact stay invalid/fail-closed.
- Legacy Physics and experimental v4 remain isolated.
- Clean-commit tests, frontend build, and scoped CI must all pass before removing draft; browser E2E must be reported separately if not executed.

## Known scientific limit

This implementation does **not** provide a certified impact transition or Gym `terminated=True`. It satisfies the bounded development demo runtime contract only, **not** the separate ambition for trainable collision terminal rewards. If PPO must learn from contact penalties, that requires an independent, Physics-certified Gym contract and a new scientific acceptance plan.
