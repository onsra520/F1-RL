# LivePose continuity & follow-camera smoothing — engineering acceptance

**Date:** 2026-11-03 (ICT). **Repository:** `onsra520/F1-RL`, branch `fix/auto-drive-lane-render-config`. **Pre-fix commit:** `bbce097fd2d87c4527e94acfe5fff7ea2403eb42`.

## Problem and changes (frontend-only)

The previous `LivePoseInterpolator.push()` switched interpolation origin to the last *received* packet rather than the pose currently displayed at the incoming packet's timestamp. Under variable WebSocket arrival cadence, this caused backward/forward car pose jumps and pulled the camera off its target, even though all Python Physics data were valid.

- `web/src/lib/livePose.ts`: on each ordered incoming authoritative frame, capture the currently sampled/displayed pose first, then blend continuously to the new authoritative pose. Keep no extrapolation beyond the latest packet, order validation, pause/resume/reset, shortest-arc heading/quaternion, and terminal authoritative pose.
- A configurable **visual speed budget**, set to 15 m/s at real-time 1x (corresponding to the frozen 54 km/h vehicle cap), extends interpolation duration for packet bursts. This is a display-only limit and never changes vehicle Physics. Incoming packet authoritative target is still reached.
- For `SIM SPEED 0.25x–16x`, the visual speed budget and camera tracking response are scaled with the selected multiplier; no forced 1x display cap in accelerated modes.
- New `web/src/lib/cameraFollow.ts` smooths chase camera position/target with frame-rate-invariant exponential damping and a bounded position change per frame, including long rendering stalls. `RacingScene.tsx` initializes the camera once on entering Drive; it never snaps the camera on each WebSocket packet.
- `App.tsx` forwards the active simulator speed multiplier to the visual system. No changes to backend Physics, hitbox, reward, PPO, source provenance, checkpoints, or model training.

## Test-first evidence

Five new continuity tests RED against the old `LivePose.ts` (packet bursts, partial blend, long gap, stale pose and quaternion). After the change:
- **Vitest full suite 68/68 PASS**. New tests cover continuity, bounded visual speed, accelerated playback, camera-step bound and convergence, abnormal delta, pause/resume, terminal, and reordering.
- **TypeScript `tsc --noEmit` PASS**, **Vite production build PASS**. A bundle chunk-size informational warning remains; build succeeds.
- **Edge E2E** Import T09 custom Bridge → Python Validate → create session (HTTP 201) → Hyper with SIM SPEED 16x → `SUCCESSFUL FINISH` after **566 steps**; **567 WS frames**, 0 reported collision, Grounded, replay timeline/export present, no JS/Console/network failures.
- No Python runtime test suite re-run for this frontend-only patch. The preceding locked backend `bbce097` already had 104/104 runtime PASS and is unchanged.

## Actual Edge rendering comparison — same Bridge/Hyper map, seed 412, 1440×900, 1x simulation, AMD Radeon 610M WebGL

| Metric (headed Edge, 50 s) | Before | After |
|---|---:|---:|
| Average rendered `useFrame` callbacks/s | 184.42 | **199.97** |
| Frame time p99 | 8.9 ms | **8.9 ms** |
| Frames over 50 ms | 5 | **0** |
| Frames over 100 ms | 5 | **0** |
| Camera–car relative shifts >0.30 m in one short frame | **32** | **0** |
| Apparent visual car speed >54 km/h intervals | 114 / 9,216 (1.24%) | **64 / 9,999 (0.64%)** |
| Largest measured camera–car relative step | 1.48 m | **0.11 m** |

The remaining 64 apparent-speed exceedances are near **55–57 km/h** rather than hundreds of km/h and are sensitive to slight differences between the in-scene `performance.now()` used to sample the visual pose and the instrumentation timestamp. In the saved sample analysis, no >60 km/h apparent-speed spike was observed at 1x. These are graphical sampling diagnostics, not certified tire velocity.

**Ramp/Hyper headed Edge (28 s):** first post-change run: 203.72 FPS, p99 8.8 ms, **three intermittent** >100 ms frames. Repeat run: **207.79 FPS**, p99 **8.8 ms**, **zero** >50 ms frames, Finish 262 steps. This is evidence of occasional browser/OS scheduling stalls, not proof that all future frame stalls are impossible.

**16x caveat:** the UI's 16x selector remained valid through a full terminal session, but measured wall-clock progress was not 16 times faster. Backend simulator throughput under available CPU load is a separate performance limitation; **do not claim 16x throughput achieved**. The visual/camera scaling and termination tests pass.

**Hardware caveat:** these measurements are Edge rendering callbacks on **ANGLE AMD Radeon 610M**, not confirmed native monitor presentation, and not the separate installed RTX 5060 Laptop GPU. Full display-native 60/120/144Hz confirmation should be performed independently.

## Acceptance

- **PASS:** frontend tests, typecheck, build, browser session/terminal/replay, car pose continuity, camera smoothing, 50-second Bridge/Hyper jitter gate, repeat Ramp/Hyper smoothness.
- **Known caveats:** intermittent isolated browser frame stalls in one Ramp run, no 16x simulation throughput guarantee, no monitor presentation-FPS certification.
- **No changes** to physics/hitboxes/PPO/training and no held-out/scientific re-evaluation.
- A release/commit is allowed **only after** the preceding tests are confirmed and git diff is clean; source was kept uncommitted throughout development.

## Evidence on the authorized local machine

`D:\F1-RL-AUTO-DRIVE-EVIDENCE-20261102\motion-red-tests.log`, `motion-final-full-vitest.log`, `motion-final-tsc.log`, `motion-final-build.log`; detailed browser JSON and image captures in `perf-headed-long-20261103` (baseline), `perf-headed-releasecandidate-20261103` (final), `perf-ramp-repeat-20261103`, `perf-16x-20261103`; terminal E2E receipt `manual-followup-20261103\motion-terminal-16x.json`.
