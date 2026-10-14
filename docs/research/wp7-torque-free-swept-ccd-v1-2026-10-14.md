# WP7 B1 — Certified CCD for piecewise-RK4 torque-free 6-DOF flight (2026-10-14)

**Research-only, opt-in; stacked on Draft PR #14. No WP7 scientific acceptance.**

## Motion model and mathematical scope

- Angular orientation and world spin are integrated at explicitly registered RK4 anchors (`TorqueFreeRotationPolicy`). The **numerical simulation's continuous pose between anchors** is defined as shortest-arc SLERP for orientation, with exact quadratic ballistic centre position and gravity-correct velocity. This is a *versioned, piecewise interpolated discrete physics model*; no guarantee of bounded error against the true continuous Euler rigid-body ODE is claimed.
- `TorqueFreeSweepPolicy` explicitly registers slice count, max shortest-arc rotation per slice, recursive CCD subdivision budget, and strict positive clearance tolerance. Uncertified rotations, ambiguous clearance, contact, precision/budget limits and accumulated energy/angular-momentum drift all fail closed.
- For every RK4 anchor interval, obstacle CCD uses exact SLERP OBB sweep with a conservative world-Y gravity sagitta (`g*h*h/8`) inflated into the rotated obstacle's body-coordinate half extents. This handles **rotated OBB obstacles** and fast mid-step translation; an actual witness contact OR inconclusive subdivision rejects the simulated action.
- Chassis–road collision uses the registered 2.5D atlas and recursive signed-plane OBB corner checks with inflated projected polygons. Per-wheel proof additionally bounds the wheel's swept projected XZ footprint via centre translation + shortest-arc angular displacement and gravity sagitta; any wheel contact or ambiguous proximity rejects. This permits safe flight near road/obstacles that the prior global spherical certificate rejected.
- Previous opt-in `BallisticFlightPolicy.torque_free` without `torque_free_sweep` retains the PR #14 conservative globally elevated/no-obstacle behavior. With `torque_free_sweep`, matching 3-axis chassis inertia is still mandatory, and `FlightStep.angular_audit` includes the globally checked anchor sequence.
- Flight continues as airborne for **full dt** only if all road/chassis/wheel/obstacle sweeps are certified clear. Scoring and pickup interactions remain disabled by the existing airborne World contract.

## Hard scientific/engineering boundaries

1. CCD certifies **the explicitly prescribed piecewise-RK4/SLERP numerical trajectory**, NOT all possible continuous-time rigid-body Euler ODE solutions between numerical integration anchors. Formal ODE enclosure and truncation error analysis are future work.
2. **Near-road contact resolution is not added**: a flight trajectory that touches, penetrates or approaches an uncertainty region of any road patch is rejected. Ground contact/impact time under this new orientation integrator requires a versioned TOI/manifold kernel and persistent grounded 6-DOF yaw dynamics (B2/B4), not approximated by the old constant-angular-velocity manifold.
3. No arbitrary track meshes, stacked road decks, tire radius/patch geometry, measured road friction, or experimental mass/inertia calibration. 2.5D height-consistent planar atlas and oriented box obstacles only.
4. Rotating-OBB near-contact conservative rejection may reduce simulation throughput. Green software regression does not prove a physically calibrated collider or scientific acceptance. No unregistered inflation tolerances/unknown motion may be interpreted as clear.
5. HELD-OUT remains SEALED. No PPO training, WP8, editing historical evidence or models, merging into recovery/main, or scientific freeze.

## Required software evidence

Exact current PR HEAD: focused B1 synthetic tests (fast tunnelling between endpoints, rotated obstacles, sloped road/body intrusions, wheel-only contact, near-road safe flight, invalid bounds and World propagation), full pytest on Windows and Ubuntu, owner-preregistered acceptance validator, Python compileall and pip check. Independently record each job log and commit SHA.

## Future B1 completion gate

Couple continuous-time torque-free quaternion ODE integration with a validated interval enclosure and contact TOI manifold; certify safety across independently registered geometry/material tolerances, incl. near-obstacle grazing and impacts. Current implementation provides a conservative, inspectable software-physics stepping-stone only.
