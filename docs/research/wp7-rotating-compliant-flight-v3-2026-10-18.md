# Rotating compliant flight with ballistic/SLERP body sweeps

Parent: bd66e495f5440f0bbcc9adbdd4bd91978b09c617, Draft PR41. Branch research/wp7-rotating-compliant-flight-v3.

## Explicit domain and runtime

`spatial_rotating_compliant_lifecycle_v3` selects the existing world-angular-momentum RK4 force field with rotating free flight, atlas-gradient phase proofs, schema8 and `spatial_rotating_compliant_lifecycle_rk4_v3`. Earlier motion/phase selectors retain their fixed airborne attitude restriction and provenance identities. Existing research registrations are unchanged; this engineering extension is not an admitted research profile.

Uncompressed springs exert zero normal/tire force. Ground actuation, boost, rolling drag and angular damping are inactive in free flight. World H is constant, orientation and angular velocity evolve using principal inertias, center motion is exact gravity and continuous unilateral contact handles touchdown. The existing energy/work and impulse audits remain in force. Runtime air validation additionally forbids actuator work, dissipation and external torque impulse. Rotation leaves subdivide while duration times the maximum of the registered angular limit and |H|/minimum inertia is >=pi; exhaustion rejects.

This remains the registered piecewise numerical trajectory, not an analytic Euler ODE clearance enclosure. Accepted anchors use shortest SLERP between their orientations and a gravity parabola for air centers. Continuous road coverage, free-extension enclosure, body road clearance, spring travel/tilt/speed/angular limits and unique monotone phase crossings still apply. Road-edge departures and unregistered collision response reject.

## Shared obstacle enclosure

The existing torque-free CCD algorithm expands each obstacle OBB along its own local axes by the gravity chord sagitta projected on those axes, then applies the existing conservative linear-center/SLERP body sweep. This calculation is extracted into `sweep_ballistic_body_against_obstacle`, reused by the previous torque-free solver and applied to every airborne leaf of the new lifecycle version. A clear expanded-obstacle sweep bounds the full parabolic body path. Contact or unresolved clearance blocks; it never fabricates an impact outcome. Earlier lifecycle versions retain their original arc checker.

## Evidence

Nine new tests pass: variable-omega world-H/energy conservation; hidden thin overhead obstacle missed by a straight chord but blocked by the parabolic envelope; public World obstacle rejection; actual Gym rotating takeoff, airborne gates, landing, recovery and finish; rotating touchdown on the actual TRAIN Ramp; no airborne ground-action/modifier effects; anti-alias subdivision and exhaustion; rejection of forged air actuator work; eight seeded comparisons against the existing independent body-Euler torque-free integrator. The broader lifecycle, contact, body collision and torque-free regression suite passes120tests in42.56s.

Independent Spec and Standards reviews report no material findings. Full exact-HEAD Windows/Ubuntu regression and hosted job receipts follow externally after commit. Readiness weights/statuses and all four scientific permissions remain unchanged. No research PPO training, HELD-OUT evaluation, scientific freeze, WP8 or canonical track edits occur.