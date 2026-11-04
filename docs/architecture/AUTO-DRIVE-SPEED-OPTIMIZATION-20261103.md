# Auto Drive spatial-speed optimization — development acceptance

**Date:** 2026-11-03 (ICT). **Working branch:** `fix/auto-drive-lane-render-config`. **Baseline commit:** `97298426fcf7c265f671d31e77725cceb623feaa`. This is a development Auto Drive-only optimization, not a change to registered Physics or frozen PPO.

## Problem and implementation

The previous controller estimated each corner's radius from individual polyline vertices and recomputed a minimum braking envelope for every future vertex. Closely sampled S-Curve navigation points and changing Racing Line offsets could cause overly conservative speed targets outside the tightest bends. Blindly raising the requested max speed or modifying Physics would not fix the root cause.

Changes solely in `src/runtime/auto_drive.py`:

- Sample the **actual authored road centreline** at approximately 2 m arc-length intervals and estimate signed-radius magnitude with a centered, 3 m chord. This suppresses polyline vertex-density noise while retaining sustained geometric bends.
- Inflate the positive curvature magnitude by a bounded conservative inner-turn radius factor for the active road-local driving offset. Do **not** use momentary lane-change target discontinuities as road curvature.
- Convert curvature to permitted planned speed by `sqrt(corner_acceleration_mps2 / curvature)`, while never exceeding the registered vehicle ground speed or UI session target.
- Preserve existing anticipation of grades, moderate/steep grade speed limits, and the real-body / cone OBB obstacle corridor's low-speed cap. Keep the previously certified road-edge limits, hitbox footprint and fail-closed contact guard intact.
- Work **backwards** from future slow sections to enforce `v_i² ≤ v_(i+1)² + 2 × max_braking × braking_factor × Δs` along sampled distances, then query a shifted, smooth interpolation for the existing physical throttle/brake controller.
- No synthetic velocity, no collision response override, no changes to checkpoint/termination semantics or frozen reward/Physics.

## Real Python Physics A/B comparison

Both runs used the **same original custom T05 S-Curve with the unchanged cone hitbox**, seed `412`. All after-change scenarios ended `SUCCESSFUL FINISH` with no runtime error, 0 reported collision and positive sample-pose OBB separation from the cone.

| Style | Preset | Original before (steps) | After optimization (steps) | Time saved | Closest accepted-frame SAT separation |
|---|---|---:|---:|---:|---:|
| Right-hand | Normal target 45 km/h | 419 | **393** | **6.2%** | ~2.17 m |
| Right-hand | Racing target 90 km/h | 419 | **393** | **6.2%** | ~2.17 m |
| Racing Line | Normal target 45 km/h | 507 | **492** | **3.0%** | ~1.38 m |
| Racing Line | Racing target 90 km/h | 507 | **492** | **3.0%** | ~1.38 m |

Recorded simulation durations on Racing preset: Right-hand 41.9 → **39.3 seconds**, Racing Line 50.7 → **49.2 seconds**. Right-hand Racing mean actual speed increased ~23.76 → **25.37 km/h**, peak 32.88 → **33.96 km/h**. Racing Line Racing mean actual speed increased ~20.36 → **20.97 km/h**, peak 30.56 → **30.86 km/h**.

**Interpretation:** Normal and Racing presets complete in equal steps in this specific obstacle-rich S-Curve: the road curvature and physical caps, not requested maximum, are dominant. Do not claim 90 km/h is physically attained or that Racing preset is inherently faster everywhere. The initial exploratory design based on raw Racing target-point curvature was rejected after yielding 700 steps; that intermediate design is **not committed**.

### Lateral-acceleration caveat

Each sample of the **planned speed envelope** obeys the user-selected lateral target `v²κ ≤ 2.2 m/s²` on its road/offset model, as checked by unit tests. Actual Physics frame speed projected onto discrete authored geometry gives a *kinematic proxy*, not a direct tire force measurement. Its maximum during Right-hand Racing was ~2.694 m/s² baseline → **2.685 m/s²** after; Racing Line Racing ~2.805 → **2.539 m/s²**. Four Right-hand and three Racing Line sampled frames on the optimized run exceed the planning target in this proxy. Thus, **do not interpret 2.2 as a proven hard bound on all transient measured lateral acceleration**; certifying that requires additional vehicle dynamics tests and/or a tighter controller margin. The optimization did not increase the peak proxy on these two scenarios.

Numeric evidence:
- [Four real after-change sessions](evidence/auto-drive-speed-20261103/speed-final-recheck.json)
- [After-change speed / lateral proxy metrics](evidence/auto-drive-speed-20261103/speed-physics-metrics.json)
- [Original baseline speed / lateral proxy metrics](evidence/auto-drive-speed-20261103/speed-baseline-metrics.json)

## Regression and scope

- Three new unit tests verify (i) speed cap and braking envelope inequality across segments, (ii) higher safe-road speed outside bends and registered physical cap, and (iii) preserved cone hitbox and grade speed limits.
- Original T05 hitbox regression tests are retained, with unchanged original fixture and registered Physics.
- **Final full runtime regression: 104/104 PASS** (434.46 s, 0:07:14) on the same final optimized source, including the original real-Physics T05 cone regression.
- Collision Physics subset: **49/49 PASS**.
- Frontend Vitest **58/58 PASS**, TypeScript typecheck PASS, production Vite build PASS. Scoped Ruff and `git diff --check` PASS.
- No PPO retrain, scientific WP3/WP7/held-out evaluation, baseline or `main` merge.

**Limitations:** No guarantee of universally faster lap times; no new obstacle search for dense clusters, no claim of real-tire contact-force certification. Source must be restarted in the running local backend before manual verification; the previously open browser may still point to a stale Python process.
