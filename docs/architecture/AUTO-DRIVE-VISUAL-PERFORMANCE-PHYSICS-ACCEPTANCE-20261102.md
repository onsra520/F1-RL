# F1-RL — WP1–WP4: visual, performance and Physics acceptance (2026-11-02)

Branch: `fix/auto-drive-lane-render-config`. Initial clean source: `0f3e2b6d6ae7a7919e2b2e2eaf3a6a322c7510ec`. This is a development/demo acceptance, not a scientific or HELD-OUT approval.

## 1. Grounded wheel alignment, pitch and roll

Ran real Python `RuntimeSessionManager` / Gym/Physics sessions, seed 412, right-hand driving style, 25 km/h requested, without retraining or altering the Physics. For each runtime frame, reconstructed the visually rendered lowest wheel point using the authoritative body position and `orientation_wxyz`. Render wheel centres are locally (±1.05, −0.07, ±1.5) m, rendered wheel radius 0.43 m. Rotated local bottom point (±1.05, −0.50, ±1.5) by the Physics quaternion and queried the nearest authoritative `TrackWorld.surface_at` patch. Gaps are signed normal distances relative to that patch, not an arbitrary ground plane. This audits visualization geometry, not tire pressure or suspension load.

| Actual scenario | Finish | Physics frames | Wheel probes | Gap abs. P95 | Gap abs. max | Max pitch | Max roll |
|---|---|---:|---:|---:|---:|---:|---:|
| Straight Sprint | PASS, 217 steps | 218 | 872 | 1.22 cm | 1.85 cm | 0.63° | 0.12° |
| Ramp | PASS, 321 steps | 322 | 1,288 | 1.34 cm | 4.39 cm | 8.93° | 0.25° |
| Elevated Turn with bridge prop | PASS, 487 steps | 488 | 1,952 | 1.61 cm | 3.09 cm | 10.70° | 0.81° |
| Supported Bridge Crossing | PASS, 745 steps | 746 | 2,984 | 2.05 cm | 3.44 cm | 10.67° | 1.67° |

**Totals:** 1,774 accepted frames, **7,096** numerical wheel/road probes; zero without road support. All four sessions finished successfully without simulator error, with zero observed airborne frames. **PASS** for these measured grounded scenarios, **not** proof of every possible surface or airborne touchdown. Source data: [physical-alignment-audit.json](evidence/auto-drive-acceptance-20261102/physical-alignment-audit.json). The Physics position and quaternion remain authoritative; no snapping or extra vertical offset was introduced.

## 2. Frame cadence, camera jitter and Hyper

Measured real Edge headless rendering over 23 s for each speed profile on the same registered `ramp-v1` track, using a DEV-only `?renderAudit=1` hook that logs actual `useFrame` callback timestamp, camera position and vehicle pose without modifying Physics. Headless Edge was initially *forced* to use software SwiftShader.

| Profile / requested target | Headless render callbacks/s | Frame time P95 | WebSocket frames | Max measured physical speed |
|---|---:|---:|---:|---:|
| Normal / 45 km/h | 26.9 | 66.2 ms | 192 | 27.7 km/h |
| Racing / 90 km/h | 30.7 | 37.1 ms | 208 | 27.7 km/h |
| Hyper / 240 km/h | 29.9 | 37.6 ms | 211 | 27.7 km/h |

All were subject to grade-based limits and real vehicle Physics. No unexpected JS/page exceptions and no instrument-detected car displacement jump over 2 m in one short frame. Software rendering **does not demonstrate a 60-FPS result**.

Also ran an additional **17 s** Hyper / Straight Sprint trial without forced SwiftShader. Edge CDP confirmed active `ANGLE_D3D11` with **AMD Radeon 610M** (`gpu_compositing` and `webgl` enabled). That trial logged **4,072 render callbacks at 239.5 callbacks/s, frame time P95 4.4 ms, zero intervals over 33.3 ms**, 165 real WebSocket frames, 27.1 km/h actual observed maximum, and the vehicle's **54 km/h** physical maximum cap. No JS/page errors or detected car teleports. **239.5 callbacks/s is NOT a measured monitor/display FPS**: a headless browser can run without display VSync, and this was not a confirmed NVIDIA RTX 5060 trial.

A high-frequency camera jitter proxy, the P95 norm of the camera's second position difference per frame after excluding first 5 s of settling, was **3.25 cm (Normal), 1.71 cm (Racing), 1.71 cm (Hyper) on SwiftShader** and **0.285 cm on native ANGLE Hyper**. The native camera-to-target chase offset P95 was ~0.62 m, expected under intentionally damped follow behavior; it is not by itself a jitter defect. This metric is not a replacement for subjective on-screen assessment.

Development-only diagnostic has been **verified absent from production Vite JavaScript**, and frontend tests/typecheck/build PASS. A Playwright recording attempt could not run because the required FFmpeg binary is missing; no video was fabricated. Real screenshots and numeric receipts are available: [software profiles](evidence/auto-drive-acceptance-20261102/perf-long-20261102.json), [native GPU profile](evidence/auto-drive-acceptance-20261102/perf-native-gpu-20261102.json), [steady camera metrics](evidence/auto-drive-acceptance-20261102/steady-render-metrics.json), [native Hyper screenshot](evidence/auto-drive-acceptance-20261102/perf-hyper-straight-native.png).

## 3. Supported vs unsupported overpass: conditional engineering acceptance

The current `e2fa09c`-based converter supports **some certified automatic separated-deck overpasses**. A physical Bridge Crossing finished 745 steps with contact geometry checked against correct-height road patches. A further converter check created safe single auto bridges with peak preview heights 4.06 m and 5.15 m. It is inaccurate to repeat the older statement that **all** automatic stacked-deck geometry is unsupported.

The converter still validates road grade, sufficient approach/runway, surface atlas and intersecting patches. Example: a bridge prop at route fraction `t=0.03` on a 40 m route was **rejected** with `unsupported_slope` and actionable instructions to move or lengthen the approach. Other unsupported surface intersections are required to fail closed, not approximated or silently fixed with Y snapping. These are **development-only** runs, not scientific route certification. Source: [physics-limits-audit.json](evidence/auto-drive-acceptance-20261102/physics-limits-audit.json).

**Engineering decision:** accept physically validated, separated, converter-supported bridges for the **development demo**. **Do not approve** arbitrary overlapping/multi-layer decks, uncertified physical contact, HELD-OUT, PPO retraining or changes to registered Physics. Such additions require separate design, tests and explicit technical approval.

## Acceptance table

| Gate | Finding | Remaining boundary |
|---|---|---|
| Grounded wheel geometry on 4 tracks | **PASS (scoped)** | No airborne contact/landing frames measured |
| Quaternion pitch/roll / road elevation | **PASS (scoped)** | Not exhaustive across all custom terrain |
| Rendering/camera cadence on 3 software and 1 native GPU run | **PASS (measured headless)** | Not a real-monitor, VSync-locked 60/120-Hz acceptance |
| Separated validated overpasses | **ACCEPT for demo** | Arbitrary stacked surfaces remain fail-closed |
| Scientific Physics / training / HELD-OUT / main merge | **BLOCKED / unchanged** | Requires separate approval |

**Validation provenance:** previously committed WP1–WP4 Python `tests/runtime` **95/95 PASS**. After introducing only the development render probe, the frontend **58/58 tests PASS**, TypeScript typecheck PASS, Vite production build PASS and production probe-exclusion check PASS. Full scientific Evaluation was **not** rerun, and no model/evidence/Physics code was modified. Work stays on `fix/auto-drive-lane-render-config` without merging. Full raw timing streams and optional diagnostic scripts remain at `D:\F1-RL-AUTO-DRIVE-EVIDENCE-20261102`.

**Unconditional all-device sign-off is still reserved** for interactive monitor/video review with measured VSync, explicit airborne touchdown cases and any separately approved advanced Physics work.
