# WP7 recovery — pilot preflight and blockers (2026-10-11)

**Scope:** Nontraining, repeatable, TRAIN-only candidate audit. This is NOT evidence that a PPO model finished any track.

## Safe entrypoint

From the recovery worktree:

```powershell
.\.venv\Scripts\python.exe scripts\wp7_pilot_preflight.py
```

The command prints a deterministic JSON object. An exit code of **2** denotes outstanding scientific review blockers, **not a software crash**. It does not write artifacts, start PPO or load models. Its status must not be described as `ready` while blockers exist.

## Pinned candidate plan

- Candidate semantics: `configs/recovery/wp7-c2-candidate.yml`, still `status: candidate`.
- Training profile: `configs/recovery/wp7-pilot-200k.yml` — 200,000 requested timesteps, 25,000 checkpoint/evaluation milestones, 8 DummyVecEnv, CPU, `MlpPolicy`.
- Pilot seed pairs: **calibration PPO/sampling (0,0)**, **independent confirmation (1,1)**. Final scientific replicate seeds are not consumed.
- Four TRAIN tracks only: `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1`.
- Preflight captures SHA256s of: candidate YAML, 200k runtime profile, base experiment, manifest, plus four TRAIN asset JSONs. World hashes remain unchanged.
- Legacy `capture_run_provenance` alone is **insufficient**, because its runtime fingerprint points to the original 50k `config.yml`. An eventual pilot executor must attach these candidate-specific fingerprints in addition to standard run/environment provenance and ensure a clean recorded Git HEAD.

## Geometric hazards to review (static, approximate)

- Right-turn: abrupt 90-degree endpoint junction; smooth traversability under the bicycle steering model has not been demonstrated.
- Slalom: tight final route junctions (~75 and ~62 degrees) and ~1.15m smallest sampled *centerline-to-obstacle-box* clearance. This is **not** a full-vehicle swept collision clearance.
- Obstacles here are axis-aligned; preflight rejects rotated obstacle boxes rather than reporting falsely precise clearance.
- Ramp and straight have no authored obstacle boxes. Static road sampling does not establish dynamic lap completion.
- `lateral_limit_meters=6` and `max_consecutive_steps=10` are candidate off-track choices, **not** approved scientific constants. Corner-cutting, high-water shortcuts and ramp launch behavior require dynamic TRAIN-only tests before actual pilot.
- Semantics and scientific freeze cannot be justified by the lost prior 4/4 results. Fresh, checksum-backed checkpoint/model evidence will be required.

## Open gates before training

1. Route feasibility / swept-vehicle clearance and shortcut exploit tests.
2. Canonical C2 experiment schema/provenance alignment without falsifying the old draft.
3. Separate **pilot executor** enforcing 25k evaluation, independent calibration/confirmation, artifact durability, Windows state-writer error handling and strict no-HELD-OUT boundary; mock-tested before any real training.
4. Only after explicitly reviewed TRAIN pilot outcomes can the new 200k candidate become frozen; WP8 remains locked.

Verification: `tests/learning/test_pilot_preflight_reconstruction.py` checks all 8 SHA256 fingerprints, exact seeds/track IDs and profile, deterministic report, named route warnings and fail-closed drift rejection.
