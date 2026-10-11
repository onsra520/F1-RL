# WP7 Reconstruction after unrecoverable local SSD

Date: 2026-10-11. Status: **reconstruction in progress**. No WP8 authorization or held-out learned outcomes.

## 1. Recovery origin

- GitHub: `onsra520/F1-RL`; remote `main` at `c0e1101c2492acbe8c11f41a07485372f99d1950`.
- Main checkout: `D:/Projects/F1-RL` (remains clean `main`).
- Isolated worktree: `D:/Projects/.worktrees/F1-RL/wp7-rebuild` on `recovery/wp7-rebuild`.
- New machine: Windows, Git 2.56, Python 3.12.15, uv 0.12.24.
- Isolated `.venv` Python 3.12.15; installed project editable with dev extras (SB3 2.9, torch 2.14, gymnasium 1.4, numpy 2.5, PyYAML 6, pytest 9).
- First clean WP6 verification: **1074 passed, 2 skipped, 157.97s**. `compileall` passed, `uv pip check` compatible.
- The old WP7 C1/C2/pilot/freeze commits were **not pushed**. In particular `e1d25723679228a295e5afa88db9185a0ffbc37c` and `33a2205bda19720d71e0ac3f7df2d9f12e03e5f6` do not exist on the accessible remote.
- No WP7 model/checkpoint/report artifacts in the new clone; `artifacts/runs` and `artifacts/evaluations` contain only README files.

This is a rebuild from existing WP6 source plus the retained technical requirements, **not a recovered byte-identical historical WP7**. New changes must use new commits and new evidence.

## 2. Immutable scientific intent

Question: Does a PPO policy with **authored route guidance** generalize to held-out geometries under the frozen environment/control/evaluation contract?

Training-only tracks (pilot): `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1`.

Held-out tracks (outcome access prohibited until separate WP8 authorization): `left-turn-v1`, `offset-slalom-v1`. Static manifest/split/hash integrity checks remain allowed.

Proposed reconstructed WP7 v1 behavior semantic IDs (subject to independent tests):

```text
algorithm   = ppo
policy      = MlpPolicy
navigation  = authored-surface-polyline-endpoint-v1
observation = rangefinders-local-route-navigation-v1
reward      = route-highwater-progress-collision-time-offtrack-v1
action      = steering-throttle-brake0-v1
race        = single-lap-ordered-checkpoints-v1
dynamics    = ground-no-slip-air-ballistic-v1
```

Pilot calibration seeds `(0,0)`, confirmation `(1,1)`, requested 200000 steps each, 25000 checkpoint cadence, PPO CPU, 8 dummy envs. Three fresh final replicate pairs reserved `(100,200)`, `(101,201)`, `(102,202)`. Do not run final replicates yet.

The old pilot showed 4/4 TRAIN for calibration and fresh confirmation at 200k, but these numbers are **historical accounts, not current reproducible evidence**. No freeze recommendation may be made based only on them.

## 3. Reconstruction gates

1. C1: Ground no-lateral-slip and physics feasibility. Start with failing focused tests; update minimal runtime; check ramp/air behavior and all existing tests. Commit separately.
2. C2: Rebuild authored surface navigation and endpoint guidance, route-aware observation/reward, off-track term, canonical six-asset content and geometry-safe routes, and telemetry/evaluator compatibility. Derive contracts from existing assets, protected split and fresh tests, not guessed historical bytes. Commit in independently tested increments.
3. Pilot: Restore training-only canonical builder, full provenance, 25k calibration ladder, the separate 175k *diagnostic* stability point, fixed 200k product candidate, fresh 200k confirmation, immutable report/gates. Never inspect held-out outcomes.
4. Re-run independent test suite and objective TRAIN-only evaluation. Preserve models/provenance/checkpoints and record SHA256s on durable storage outside ignored artifacts.
5. Only after fresh evidence supports 4/4 both runs, perform a separately reviewed protocol freeze: canonical 200k profile, final seed arrays, `ExperimentDefinition.require_frozen()`. Then request distinct WP8 execution approval.

## 4. Constraints

Do not push, merge, force rewrite history, silently tune PPO/rewards, train on held-out, start final replicate seeds, or claim WP8 readiness. Stop/review any new algorithmic change not justified by regression tests. Keep the existing `D:/Projects/.worktrees/TroOi-Project` and other projects untouched.
