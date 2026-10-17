# Bounded atlas-gradient lifecycle phase crossings

Parent: `bf94ba5681fc2a1adad44d1a5d17c359e1acf7ad`, Draft PR40. Branch `research/wp7-atlas-phase-lifecycle-v2`.

## Runtime change and versioning

`spatial_compliant_atlas_lifecycle_v2` explicitly selects `atlas_gradient_v2` in the existing compliant integrator. Physics schema7 and `spatial_compliant_atlas_lifecycle_rk4_v2` identify the new domain. World, PhysicsConfig state/track/transition validation, Gym and factory action guards recognize it. Existing registrations continue selecting v1 and retain their exact physics hash. No automatic migration, new force law, calibrated parameter or scientific permission is introduced.

The previous unique-crossing proof required all potentially crossed patches to lie on one horizontal plane. The new version permits continuous 2.5D sloped and mixed-normal patches when every potential wheel/patch gap has the same provable monotone direction. Fixed airborne attitude, continuous road atlas coverage, spring travel, body/obstacle sweeps, work/impulse audits and unsupported collision recovery restrictions remain in force.

## Numerical trajectory proof

An accepted contact leaf uses linear center interpolation and shortest quaternion SLERP. For local wheel reference r and plane height gradient a=n/n.y, the normalized wheel/plane gap has second derivative bounded by theta^2 |r| |a|. Its derivative differs from its secant by at most half that bound. Each potentially crossed plane must satisfy the common phase-directed secant inequality, with an additional conservative coordinate-dot roundoff allowance for atlas-v2. The wheel footprint enclosure includes the entire center chord plus rotational arc bound. Existing atlas construction requires continuous heights at seams, so piecewise gap derivatives share one strict sign across all possible patch transitions. Every wheel gap is then monotone; the aggregate compressed/free state has at most one threshold crossing in the accepted <=1e-8s phase bracket.

This is a proof for the registered piecewise numerical trajectory, not an enclosure of an exact calibrated ODE. The new roundoff allowance only tightens atlas acceptance. Horizontal-v1 retains its original comparison and identities. Unresolved direction, incomplete coverage, rotating free flight or incompatible atlas geometry still reject.

## Runtime evidence and review

Failing-then-passing tests cover the new kernel selector and public PhysicsConfig boundary. Eight new tests cover actual TRAIN Ramp touchdown, supported recovery/passive work, symmetric two-normal contact with both patch forces, public Simulation, actual Gym/owner outcome with valid observation, unknown selector rejection, the old owner profile's exact physics SHA and rejection of an audit labeled as the old solver. Existing hidden narrow-ridge and dual-valley regressions run under both phase selectors. Canonical track files are unchanged; the Gym landing fixture explicitly starts at its test landing location.

The focused lifecycle/contact/race/Gym suite passed82tests in37.25s before the final roundoff-only tightening; all8new tests passed afterward. Independent specification and standards reviews report no material findings. Full exact-HEAD regression and compatibility evidence are recorded externally after commit.

Readiness weights/statuses and the owner execution registration remain unchanged. Broader rotating flight, road-edge release, generalized landing/contact robustness and independent physical/scientific validation remain incomplete. No research PPO training or HELD-OUT evaluation is performed.