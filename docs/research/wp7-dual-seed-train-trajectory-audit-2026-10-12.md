# WP7 recovery: dual-seed real PPO TRAIN trajectory audit (2026-10-12)

## Decision: `review_required` — do not scientific-freeze or open HELD-OUT

**Evidence:** Original independent 200k exploratory PPO runs on the four TRAIN tracks.

| Seed pair | Actual timesteps | Final TRAIN success | Earliest 4/4 milestone |
|---|---:|---:|---:|
| Calibration (0,0) | 200,704 | 4/4 | 175,000 |
| Confirmation (1,1) | 200,704 | 4/4 | 50,000 |

Each milestone report covers **one deterministic episode per track**, not repeated stochastic trials. Both original `evaluations/manifest.json` passed the existing SHA256 verifier. Their source fingerprints match the present candidate, 200k profile, experiment and TRAIN assets; original weights and reports were not changed.

## Engine limitation discovered in code review

- `TrackWorld.query_point()` calls `volume.contains(position)` using the vehicle **centre point**.
- `step_world()` queries obstacles at each candidate position after integrating `dt=0.1 s`; a thin obstacle crossed between samples may escape collision detection.
- `VehicleParameters` has a 3.0 m **wheelbase**, but no defined physical body length, width, corner radii or collision hull; therefore neither point collision count 0 nor a 2D centreline buffer certifies full swept-vehicle safety.
- The recovery candidate permits `6 m` off-track lateral distance with truncation only after `10` consecutive over-limit samples. Passing race checkpoints and finish may not imply following the authored polyline closely.

## New independent read-only audit

`f1rl/learning/trajectory_audit.py` and `scripts/wp7_audit_train_trajectories.py` reload each **archived real final PPO ZIP** (CPU), rebuild **exactly** the four TRAIN scenarios, replay the deterministic episode, and compare finish status / simulator obstacle events / completion time with the archived final milestone report.

For every pair of successive 3D centres, the audit finds the **exact 2D XZ segment-to-obstacle axis-aligned rectangle distance**, including the segment interior between simulator samples. It also measures max XZ distance to the authored route and longest simulator displacement per step.

**Method limitations:** XZ ignores height and can flag safe overhead passage as near-obstacle. Only the centreline is tracked; 0.5 m, 1.0 m, and 1.5 m are **hypothetical radial-proxy probes**, *not* a measurement of the physical car. Obstacles in this reconstruction are axis aligned; rotated obstacles cause this audit to reject. No HELD-OUT environment is created.

| TRAIN track | Calibration max deviation (m) | Confirmation max deviation (m) | Calibration swept XZ obstacle gap (m) | Confirmation gap (m) |
|---|---:|---:|---:|---:|
| straight-sprint-v1 | 1.352062 | 1.338830 | none | none |
| slalom-v1 | 2.999362 | 1.890345 | **2.024188** | **1.951178** |
| right-turn-v1 | **5.511796** | **4.415475** | none | none |
| ramp-v1 | 1.398490 | 1.261880 | none | none |

- Both final real policies replayed **4/4** on TRAIN, with no engine-reported obstacle collisions.
- Slalom centreline does not intersect any obstacle in the continuous XZ audit. Both have >1.5 m planar clearance to obstacle rectangles. This **does not certify body footprint**.
- Right Turn is the principal route-following risk. The maximum lateral errors of 5.51 m / 4.42 m are under the configured 6 m off-track threshold, but may represent significant corner-cutting. A review must decide acceptable route corridor based on design intent **before** using HELD-OUT.
- Maximum per-step centre displacement was ~1.43 m (calibration) and ~0.81 m (confirmation), so testing only post-step point containment is insufficient as a general collision proof.

## Evidence location and reproducible command

Original model ZIPs and eight milestone JSON reports per seed remain in:

`D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-11_205037_ppo-multitrack-seed0-sample0`

`D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-12_203835_ppo-multitrack-seed1-sample1`

New *separate* audit JSON:

`D:\Projects\.archives\F1-RL\wp7-exploratory\audits\dual-seed-train-trajectory-audit-v1.json`

Its SHA256 when originally produced:

`0e44bcaf738f5c4eadb2f726b3a8108af64d1690f7a9fbf28ccbdabcd771d188`

The audit JSON records original model SHA256 and manifest SHA256, source Git revision, replayed TRAIN outcome, per-track geometry and explicit limitation notes. Command:

```powershell
cd D:\Projects\.worktrees\F1-RL\wp7-rebuild
.\.venv\Scripts\python.exe scripts\wp7_audit_train_trajectories.py `
  --run 'D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-11_205037_ppo-multitrack-seed0-sample0' `
  --run 'D:\Projects\.archives\F1-RL\wp7-exploratory\2026-10-12_203835_ppo-multitrack-seed1-sample1' `
  --output 'D:\Projects\.archives\F1-RL\wp7-exploratory\audits\dual-seed-train-trajectory-audit-new.json'
```

The CLI rejects existing output paths and all paths inside the Git worktree; it **does not** delete or overwrite historic evidence.

## Remaining review gates

1. Specify vehicle 3D footprint and a collision hull, then add **continuous swept-volume** validation (3D, including ramp elevation), benchmark against original point-engine outcomes. Any geometry change requires a new experiment candidate and new training protocol; do not rewrite old 4/4 scores.
2. Review Right Turn's physical turn feasibility, nominal course corridor and allowable corner-cutting; pre-register route-deviation checks before any held-out evaluation.
3. Review stochastic policy robustness and seeds/reset variations. A deterministic single replay is a provenance check, not an independent robustness replicate.
4. Resolve run provenance missing from runner (`state.provenance_file = null`): record Python/library versions, exact PPO config, seeds, Git source identity and whether model restarts/weights correspond to the run.
5. Preserve model/report/audit artifacts on an **independent storage device or remote backup**; `.archives` still shares a physical drive with other project data.
6. Only after review and explicit frozen protocol authorization, consider opening HELD-OUT; do not use HELD-OUT results to tune the model.

No HELD-OUT rollout, re-training, new scientific seed replicate, WP8 or `main` merge was performed by this audit.
