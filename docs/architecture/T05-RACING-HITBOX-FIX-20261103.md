# T05 — Racing Line obstacle hitbox regression and fix

**Date:** 2026-11-03 (ICT). **Branch:** `fix/auto-drive-lane-render-config`. **Baseline:** `abd2c5035e87f2f633a3c45bfbd3a80fa0aa6875`.

## Original user-visible failure, captured before source changes

- Custom track: user-exported `my-first-circuit.json`, SHA-256 conversion ID `3359bae959535baa40d337ae7677540ab64f022e739acd148fa2f9b65369ec37`. The original exported map is preserved as `tests/runtime/fixtures/t05-original-obstacle-track.json`. An older replay's embedded preview has a normalized generated ID and is not byte-identical to the original editor draft; the fixture hash was independently checked.
- T05 original: Racing Line / Racing, seed 412, finished **FAILED at step 263**, `FlightPhysicsError: flight/substep body hit obstacle: no scored impact response registered`. Same failure was observed by the user with Racing Line / Normal, but not Right-hand / Normal and Racing.
- Source Physics registered hitboxes: `VehicleBodyGeometry(width_m=1.8, height_m=1.2, length_m=4.0, local_center=(0,0.3,0))`; cone `cone-001` is an oriented box with half-extents `(0.65,0.60,0.65) m` at `(14.5226,0.66,21.9052) m`. **These dimensions have NOT been changed.**
- At replay tick 263 the car's position was ~(12.577,0.500,18.785) m, the Racing Line target lateral offset was ~2.2 m and the cone was ~2.8 m from the reference centerline. The previous accepted body's OBB is separated from the cone OBB by a positive SAT projected gap ~0.748 m. The next continuous Physics interval detected an obstacle contact without a registered *scored impact response*. A positive gap at the previous sampled frame does not disprove continuous swept contact; nor does `COLLISIONS=0` certify no contact.

## Confirmed implementation

Change is confined to development Auto Drive controller `src/runtime/auto_drive.py` and session-to-controller constructor binding `src/runtime/session.py`. **No changes** to registered Physics, contact sweep, scoring, reward, body/collision volumes, model/checkpoints or training evidence.

The session passes the authoritative `track.obstacles` and registered `physics_config.body_geometry` to the controller. Racing Line now creates an obstacle corridor using actual `OrientedBoxVolume` size/orientation, projection onto the road-local lateral normal, registered chassis dimensions, vehicle/yaw clearance and a planning margin. The planner blends the intended Racing offset into a safer offset starting 36 m before the hazard and recovers smoothly after passing it. It reduces real physical target speed near the hazard (without synthetic velocity), enforces the usable road-width margin, and **rejects** a Racing setup if no conservative collision-free lateral corridor fits. It ignores obstacles on sufficiently separated vertical decks. The original right-hand/centerline planning logic remains unchanged. Physics still owns SAT and continuous collision certification; unscored impacts remain invalid.

**This is not a general global path planner.** The tested fix covers the original isolated cone and safe lateral passage, not simultaneous/dense obstacles, arbitrary geometry, or a certified crash impulse solver. No "reduce cone hitbox" shortcut was used.

## Original-map development replays after fix — seed 412

| Style | Preset | Result | Steps | Nearest accepted-pose hitbox separation (SAT projection bound) |
|---|---|---|---:|---:|
| Racing Line | Normal target 45 km/h | **PASS — Successful Finish** | 507 | 1.385 m |
| Racing Line | Racing target 90 km/h | **PASS — Successful Finish** | 507 | 1.385 m |
| Right-hand | Normal | **PASS — Successful Finish** | 419 | 2.145 m |
| Right-hand | Racing | **PASS — Successful Finish** | 419 | 2.145 m |

These four are actual seed-412 Python Gym/Physics replays using **the exact original custom track hash** with the cone present. The closest hitbox separations are lower bounds on sample-pose OBB separation and must not be interpreted as a guarantee for all future trajectories. The user-visible benchmark T05 prior to the fix **FAILED at step 263** with the cone present; do not overwrite that historical failure report.

Evidence: [before audit](evidence/t05-hitbox-20261103/t05-hitbox-before.json); [after rechecks](evidence/t05-hitbox-20261103/t05-fixed-recheck.json). Full local replay captures and command logs are retained under `D:\F1-RL-AUTO-DRIVE-EVIDENCE-20261102` and the originally downloaded failed replay under the user's Downloads folder.

## Frontend mesh versus registered Physics hitbox

A numerical cross-check using compiled frontend `pointAt` on the original exported editor map gave cone visual center (x,z)=(14.5449,21.9196) m and visual center Y=0.68 m, while the authoritative Physics OBB center is (14.5226,21.9052) m and Y=0.66 m. Horizontal difference is **0.0266 m** and yaw difference **0.484 degrees**. This small mesh/physics placement difference does not explain the impact; the racer target came within the actual combined OBB clearance. The body Physics OBB is not identical to its decorative visual chassis (1.8 m versus 1.5 m rendered body width), hence the controller must plan with the **Physics** hull, not the mesh. [Frontend alignment receipt](evidence/t05-hitbox-20261103/t05-visual-vs-physics-hitbox.json).

## Regression gate

- `tests/runtime/test_auto_drive.py`: unit checks for real hitbox use, smooth anticipation, right-hand behavior invariance, different deck and blocked/narrow road rejection.
- `tests/runtime/test_t05_hitbox_regression.py`: test both Normal and Racing presets with original user map, seed 412, real Physics to Finish, positive SAT hitbox clearance, no new Physics exception.
- The original `src/simulation/world.py` fail-closed `verified_obstacle_contact` guard is unchanged.
- No change to frozen PPO model, WP3 provenance or scientific/HELD-OUT gate.

**Gate status: PASS for this T05 engineering fix.** Scoped Auto Drive unit tests **19/19 PASS**; complete `tests/runtime` regression with the two actual T05 seed-412 tests **101/101 PASS** (383.89 s); collision Physics subset (`tests/sandbox/test_body_collision.py` and selected WP3 impact/certification tests) **49/49 PASS**; frontend Vitest **58/58 PASS**, TypeScript typecheck PASS, Vite production build PASS; source Ruff lint PASS. The complete 3,000+ test research suite was not rerun for this scoped patch. Browser UI on port 5173 was running an older Python backend process during the patch; **restart services intentionally before interactive T05 manual retest**. Do not claim the existing browser process automatically acquired the updated backend controller.

**Acceptance:** This fixes the **T05 Racing Line + isolated cone scenario**, with hitbox-aware avoidance in Auto Drive. General impact scoring and other protected Physics work remain distinct tasks.
