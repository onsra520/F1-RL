# T6 Release Acceptance — Runtime Simulator (2026-11-04)

This is a **development/demo runtime acceptance**, not WP7/HELD-OUT scientific admission or authorization to retrain PPO.

## Pinned source and full regression

- PR: https://github.com/onsra520/F1-RL/pull/56 (fix/auto-drive-lane-render-config onto feature/runtime-simulator-bridge).
- Release code commit: `adfb33ec505c22cd4dab18d6077dbd5e5d540367`.
- Exactly the same Git tree as the separately tested T5 clean candidate `8f172f5bcad449b458790b13f32024474b698fc7`: `a068bdd3e82a31c669f95febfee59284b045baae`.
- T5 local runtime source SHA-256: `3282c7339f0a60cfdeb6031e207eacfd739a36aea68fc3d0ff850d070467a87b` (local bytes; Git tree SHA is platform-portable).
- Training provenance `fe83f51` retained unchanged; no PPO retraining or held-out use.
- One full clean-source, sequential Python regression: **3,214 collected; 3,211 PASS; 0 FAIL; 0 ERROR; 3 Windows SKIP; exit 0**. Duration 1,820.58 seconds.
- All 18 previously failing/timeout cases rerun serially: **18/18 PASS**. WP2 clean-source contract: **11 PASS**.
- Original manual development acceptance: **18/18 PASS within specified scope**. Frontend Vitest **77/77 PASS**, production build PASS.
- Three Windows SKIPs check anti-symlink-escape but the host lacks symlink privileges (WinError 1314). They are NOT counted as PASS.
- Frozen T5 manifest SHA-256: `bc18b285f0d7091b0899df0d7c21de78fe3aa035713e57da94d2a075704bbbdd`.

## T6 manually retested limitations

**L03 — PASS:** Two independent headed Edge sessions on the identical source-bound track and seed 412 reached successful terminal, not merely a sampled partial run. Centerline 160 Physics frames, Racing 175; both replay trace hashes matched. 70 matched longitudinal-position samples show path differences (mean absolute lateral difference 2.162 m); 0 browser errors.

**V03 — PASS within certified support envelope:** Four valid separated 3-tier road layouts (original, rotated, mirrored, scaled) with deck heights 0/2.4/4.8 m and Physics steps PASS; two unsafe layouts rejected fail-closed. Three pinned regression tests PASS. Previously a real 1,192-tick Physics session completed successfully and the corresponding three levels were displayed in headed WebGL replay. This does **not** certify every arbitrary overlapping multideck geometry.

**S02 — PASS for a bounded load on NVIDIA RTX 5060:** Headed Edge WebGL 1440×900, simulation speed 1×, four sequential 15-second trials with 0, 8, 16, and 24 CPU workers at 85% duty cycle. WebGL identified NVIDIA RTX 5060 via the per-browser `--force_high_performance_gpu --use-angle=d3d11` option. All four runs recorded **0 render intervals over 50 ms**; the worst intervals were 7.6, 12.6, 7.9 and 8.0 ms. The integrated AMD 610M achieved 1/4 strict PASS at 1440×900 and 3/4 at 960×600. Render callback counts are not display/VSync FPS; Physics WebSocket throughput still decreases under CPU contention. Unbounded CPU/GPU loads are not certified.

- T6 final scoped manifest SHA-256: `4202222f49a663489224e9581ece5c560fe85e184709f953529f525a0aa160e6`.

## CI and release decision

The T6 GitHub Actions workflow validates the pinned T5 code against the release commit, portable Windows Python runtime/track/Physics tests, and frontend Vitest/build. This hosted subset does not replace the separate evidence-dependent full 3,214-test T5 regression.

**Do not merge unless actual GitHub Actions checks succeed and independent review completes.** Absence of checks must never count as PASS.

Large JUnit, traces, screenshots, and hashes remain off Git under `D:\F1-RL-AUTO-DRIVE-EVIDENCE-20261102\T5-FULL-REGRESSION-20261103` and `D:\F1-RL-AUTO-DRIVE-EVIDENCE-20261102\T6-CAVEAT-RESOLUTION-20261104` on the authorized Windows machine.