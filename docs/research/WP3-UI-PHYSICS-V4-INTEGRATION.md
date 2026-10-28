# F1-RL: Racing Sandbox + Physics v4 integration

This integration joins the browser Racing Sandbox from frontend/racing-sandbox at
49a5819f20c42805e0aa729c5b9d18025f9bd1c0 with the source of the
completed WP3 TRAIN-development Physics v4/Evaluation from
fix/wp3-road-impact-lifecycle-20261027 at
6c479540134d98d7d27331879c926c8ba82c2965.

## Verified boundaries

- The browser editor and manual drive remain visual-only. Imported PPO Replay
  files use f1rl_racing_sandbox_replay_v1. The frontend does not yet have a
  Python TrackWorld compiler, API/WebSocket bridge, or native FULL-06 JSONL
  replay adapter. A successful code merge does not imply live UI/physics wiring.
- Python remains authoritative for Physics v4, registered crash/tilt outcomes,
  checkpoint scoring and development evaluation.
- No PPO training, sealed HELD-OUT access, or WP7 scientific acceptance was
  introduced by the merge. Training models/source and older evaluation
  attempts remain unchanged.
- Original FULL-06 evidence is intentionally **not copied into main**:
  [FULL-06 raw evidence and reports](https://github.com/onsra520/F1-RL/tree/evidence/wp3-issue51-eval-only-20261024/docs/research/evidence/wp3-issue51-full06-traindev-20261027).
  RUN-ID WP3-OWNER-TRAINDEV-20261027-FULL-06: 96/96 cells, 5760/5760
  valid episodes, 1550 finishes, 0 invalid, independent evidence audit PASS.
- The incoming main-regression.yml GitHub Actions workflow was excluded
  from this integration. Existing main workflows were not removed or changed.

## Local checks

Run these commands from the repository root:

    cd web
    npm ci
    npm test
    npm run build
    cd ..
    python -B -m pytest -q tests

The main integration can be used for the next phase of **implementing**
a typed Python track-to-runtime compiler and real PPO Replay transport.
It is not a scientific certification or a claim that PPO finishes every
track (Right Turn has no FULL-06 finishes).
