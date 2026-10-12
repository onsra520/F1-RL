# WP7 — Independent 3D swept-body proxy audit (2026-10-12)

## Decision

**REVIEW REQUIRED; no scientific freeze.** The two original 200k real PPO models still succeed 4/4 on their four **TRAIN** tracks under the existing point-based simulator. This new audit checks three **hypothetical** 3D vehicle shapes, but it does not determine true vehicle geometry or validate a continuous full-body physics simulator. No HELD-OUT data, model training, or change to original evidence occurred.

## Provenance and source boundaries

| Role | Run ID | Actual PPO steps | Archived TRAIN result |
|---|---|---:|---:|
| Calibration (ppo=0, sampling=0) | `2026-10-11_205037_ppo-multitrack-seed0-sample0` | 200704 | 4/4 |
| Confirmation (ppo=1, sampling=1) | `2026-10-12_203835_ppo-multitrack-seed1-sample1` | 200704 | 4/4 |

Source baseline was committed clean at `5cfc5674e9a6547fadc2a203fcf5fa210b739f68`. Each run's original eight milestone reports, manifest hashes, final model SHA256 and training-source fingerprints were verified before replay. The audit compares each deterministic TRAIN replay with the archived finish, obstacle-hit count and completion time. Only four `TRAIN_IDS` are instantiated.

New independent audit JSON (outside source Git and outside historic run directories):

`D:\Projects\.archives\F1-RL\wp7-exploratory\audits\dual-seed-swept-3d-body-proxy-v1.json`

SHA256: `da1accdd0d0f5bdafb7e790466f8c30a6a054eb637b9c1dc6820b2114e2bfed8`.

## Geometry method and validity

- Exact Euclidean **3D** closest distance between a vehicle centre's finite step segment and an **oriented obstacle box**, found by minimizing piecewise quadratic distance between box slabs. Obstacle rotation and vertical clearance are included.
- The *hypothetical vehicle* is an oriented rectangular box in each sampled pose, using 9 longitudinal stations. Its cross section and half-station spacing are enclosed conservatively by spheres.
- During each timestep, the calculation assumes **linear vehicle-position interpolation** and **shortest-arc quaternion slerp**. For each sphere station, the rotational arc-to-chord deviation is conservatively bounded by `|local_station_vector| * (1 - cos(relative_rotation_angle/2))`; this plus the station's covering radius is subtracted from exact station-chord-to-obstacle distance.
- **Positive signed lower bound** means the assumed proxy box is separated from the obstacle **only under the interpolation assumption**. **Zero or negative lower bound** means **inconclusive**, NOT actual collision. This calculation does not substitute for a vehicle-specific swept OBB collision engine or a physically validated inter-step trajectory.
- No body width, length, height or body-centre offset was fixed in the original vehicle contract (only wheelbase=3.0 m and ride_height=0.5 m). The three dimensions below are **sensitivity probes**, not measured car dimensions.
- Because only `slalom-v1` has obstacles among the four TRAIN scenarios, the obstacle-clearance metric is driven by **Slalom**; other tracks have `null` / not applicable for obstacle-gap checks.

## Result: worst conservative obstacle gap (metres)

| Assumed body L×W×H (m) | Assumed centre offset upward (m) | Calibration seed 0/0 | Confirmation seed 1/1 | Interpretation |
|---|---:|---:|---:|---|
| 3.2 × 1.4 × 1.0 | 0.45 | +0.789700 | +1.067034 | Both pass assumed proxy |
| 4.0 × 1.8 × 1.2 | 0.55 | +0.403928 | +0.840248 | Both pass assumed proxy |
| 5.0 × 2.2 × 1.4 | 0.65 | **−0.040417** | +0.598965 | Calibration **inconclusive**, confirmation passes proxy |

Both still replayed 4/4 TRAIN and zero engine obstacle-hit events. The large-profile negative bound does not establish actual clipping. Note the previous XZ centre-path geometric clearance was 2.024188 m vs 1.951178 m; the new, more useful body-oriented 3D conservative metric reverses which seed has more margin, because each policy's orientation and swept body shape matter.

The previous trajectory audit also found Right Turn maximum horizontal route deviation of **5.511796 m** (seed 0/0) and **4.415475 m** (seed 1/1) against a configured 6 m offtrack tolerance. **Right Turn route validity is still unresolved even though it has no obstacles.**

## Scope and review gates

- Pure checker: `f1rl/learning/swept_volume_audit.py`.
- CLI: `scripts/wp7_audit_3d_swept_volume.py`. Re-running must use a new report filename; the original JSON is immutable by design.
- Unit/invariant coverage: `tests/learning/test_swept_volume_audit_reconstruction.py`. Covers tunnelling, 3D overpass, rotated boxes, endpoint reversal and body-rotation interpolation bound.
- The replay intentionally inspects the existing immutable `env._simulation.snapshot.world.vehicle` orientation because public environment telemetry exposes position/velocity but not quaternion; this is **read-only** but coupled to a private simulator attribute.

**Still required before any scientific freeze or HELD-OUT opening:**
1. Establish actual body geometry from intended vehicle specification, collision-hull position relative to chassis and tire placement.
2. Define acceptable Right Turn corridor and corner-cutting metric independent of outcomes.
3. Review and if necessary replace the simulator's point obstacle checks with a true body-aware continuous 3D collision method **in a new scientific candidate, with new training**. Do not reinterpret existing 4/4 results as performance under changed physics.
4. Address `provenance_file=null` by registering exact run environment, dependencies, source and model archive identity.
5. Confirm independent model/report backup and seed/replay robustness under a pre-registered evaluation protocol.
6. Keep HELD-OUT `left-turn-v1` and `offset-slalom-v1` untouched until rules and candidate are frozen.

No HELD-OUT rollout, re-training, WP8, or merge to `main` in this audit.
