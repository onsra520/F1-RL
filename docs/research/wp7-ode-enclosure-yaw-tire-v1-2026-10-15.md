# WP7 blocker reduction: analytic Euler ODE clearance, grounded yaw, tire/calibration contracts
**Date:** 2026-10-15. **Research only; stacked on Draft PR #15.** HELD-OUT remains sealed.

## B1: Analytic envelope distinct from RK4 interpolation

`euler_ode_clearance.py` derives an angular speed upper bound from constant unforced rigid-body rotational energy and the smallest registered principal moment of inertia:

`omega_bound = sqrt(2*E_initial / min(I_pitch, I_yaw, I_roll))`.

Every material point of the **exact torque-free Euler ODE** lies within `r * min(omega_bound * t, 2)` of the initial-world attitude contribution, for a point at radius `r`, assuming perfectly torque-free motion, fixed positive principal inertias and exact initial state. The centre travels the exact ballistic parabola. At every recursively bounded time interval, the certificate combines (a) midpoint centre translation and gravitational sagitta, (b) angular excursion from **the exact initial quaternion** (not RK4 anchors), (c) OBB SAT separation lower bound, (d) swept chassis/road footprint and (e) all four wheel/road separations. A safe result is an analytic enclosure; uncertainty returns the earliest unresolved time bracket, NEVER a fictional impact observation. This is conservative and gets loose for large `omega*dt`.

`TorqueFreeSweepPolicy.analytic_ode` is disabled by default. If explicitly supplied, this analytic clearance verdict MUST be clear *before* existing RK4 anchor/SLERP collision certification may advance. This does not certify a numerical collision impulse or actual continuous ODE trajectory state from RK4: Euler energy drift/trajectory error and a true continuous time-of-impact solver are separate acceptance tasks. In particular, an uncertainty interval is **not** a resolved impact time.

## B2: Opt-in grounded dynamic yaw

`grounded_yaw_dynamics.py` supplies a small-angle/slip **planar dynamic-bicycle** yaw integrator. Wheelbase axle geometry, chassis mass, registered yaw inertia, steering, axle cornering stiffness, friction capacity, maximum yaw rate, minimum speed, slip-angle domain and force audit are explicit. Front/rear lateral forces are friction capped `|Fy| <= mu * N_axle`; steering produces an axle moment about world Y; nonzero incoming yaw spin is preserved. Below low-speed threshold, above slip/spin limit, or incoherent parameters, fail closed; do not jump to the old kinematic yaw.

`step_world(..., grounded_yaw_policy=...)` is opt-in, requires independent suspension, forbids simultaneous automatic takeoff and enforces mass/axle length/flight manifold inertia consistency. It supplies a longitudinal legacy drivetrain candidate to the yaw module, then lets the existing independent-suspension solver certify every chassis/wheel substep. This is not full 6-DOF tire force integration; no separately calibrated longitudinal slip or transient load transfer.

## B3 and B4: Wheel radius/contact vs suspension references

`tire_calibration_contract.py` introduces an explicit four-wheel HUB geometry, tire radius and width; a spherical-wheel-to-inclined-plane geometric candidate computes signed surface gap and 3D normal contact point. It does NOT silently migrate the legacy wheel suspension points to hubs, assert finite patch coverage, or integrate sphere/tire forces into the existing PGS manifold. An actual tire-patch or compliant lateral force model still requires a distinct migration/review. Mixed-deck contact, grazing manifolds and PGS complementarity remain explicitly unaccepted.

## B5–B7: Acceptance evidence still missing

`MeasuredPhysicsParameter` requires name, unit, measured value, absolute uncertainty, evidence ID and method, and `PhysicsCalibrationRegister` requires an independent verifier ID. The required WP7 measurement checklist includes mass, 3 inertias, radius, width, springs, damping, axle tire cornering stiffness, grip and restitution. The registry does not invent measurements or infer calibration from synthetic tests.

Scientific acceptance still needs:
- independently verified measured material/geometry values with source traceability and uncertainty;
- preregistered physical validation datasets, independent reviewer, and reproducible analytic-vs-reference comparisons;
- registered robustness protocol with 60+ stochastic episodes per policy x TRAIN track x stratum, 3+ paired seeds, one-sided 95% CP lower bound >= 90% in every required cell;
- immutable evaluation provenance, validated evidence backup/restore on private Google Drive, strict HELD-OUT seal and owner decision.
- WP8 prohibited until scientific freeze; no HELD-OUT, no PPO training, no historical model/evidence changes and no merge to recovery/main from this PR.

## Required CI

Windows and Ubuntu exact-HEAD focused unit/integration tests for analytic enclosure (safe/unsafe, tunnelling), mass/axle/force-limited yaw World transfer, tire hub/normal contact and calibration evidence; full pytest; preregistration validator; compileall; pip check. Green software tests do not represent verified physical calibration or WP7 scientific acceptance.
