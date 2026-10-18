# Ramp and mixed-normal recovery acceptance

This inventory closes software traceability for the existing bounded lifecycle implementation. It does not register new physics, alter weights, certify arbitrary contact geometries, or grant scientific permission. Every capability keeps its original denominator: bounded_lifecycle45, ramp_domain30, mixed_normal_recovery25.

## Scope and identities

The canonical TRAIN Ramp world remains `775f795aa2763f64916e68e6a64fc763021e875c384050a974c747b6b4907978`. The explicit `spatial_rotating_owner_v3` profile retains physics `fe6e753f1471e83f8103bae5ad64e050d316c3ddb9186c51cba3c27a3e8ddf5b`. New recovery fixtures use exactly those physical parameters and canonical track, with the existing vehicle-position spawn selector instead of the registered horizontal-road spawn selector. Fixture physics is separately pinned to `a812522f2d60dad4059dcf735e6ec88785fb75915280bad213a0899f79f4a87e`. The complete documents are compared excluding only this declared spawn selector.

The fixtures start consistently airborne above the incline at z80m, or straddling the canonical approach/incline seam at z50m. They use explicit pitch -8/-4degrees, downward velocity1m/s and zero controls, dt0.01s for50ticks. Navigation starts at the declared fixture position; TRAIN randomized reset is disabled. This is a public Gym integration fixture, not a registered reset, PPO episode or robustness claim. The registered ingress continues to require horizontal road spawns; no production guard is changed.

Supported numerical dynamics use continuous road atlases with complete wheel-footprint coverage, consistent persistent pose/suspension/flight state, bounded tilt/travel/rotation and bounded event/root refinement. Registered limits remain max tilt0.2rad, angular speed10rad/s, bump/droop0.12m, atlas seam tolerance1e-6m and20 energy-refinement levels. Airborne intervals use world angular momentum and gravity with zero ground actuation/dissipation. Contact is unilateral and passive except declared drive/brake work. Unsupported geometry, inconsistent states or unresolved bounds reject rather than manufacture a transition. Scientific calibration remains independent.

## Capability evidence

| Existing capability | Positive and integration evidence | Negative and boundary evidence |
|---|---|---|
| bounded_lifecycle,45 | existing `test_pv5_lifecycle.py`; rotating Gym takeoff/airborne gates/landing/finish in `test_spatial_rotating_lifecycle.py`; actual four-track finish/stop certificates | existing historical physics, transition forgery, phase and action-contract rejection tests |
| ramp_domain,30 | new `test_canonical_ramp_recovery.py` incline case: actual Gym touchdown and50tick persistent grounded recovery; canonical Ramp traversal in `test_wp7_rotating_owner_execution.py`; independent sloped dynamics in `test_spatial_atlas_lifecycle.py` | new historical-horizontal-selector rejection; hidden ridge/valley cases in `test_spatial_compliant_lifecycle.py`; body/ballistic obstacle checks in `test_spatial_rotating_lifecycle.py` |
| mixed_normal_recovery,25 | new canonical approach/incline seam case: touchdown/recovery with simultaneous positive force on distinct canonical normals; symmetric mixed-normal landing in `test_spatial_atlas_lifecycle.py`; actual mixed-normal impulses in `test_spatial_contact_dynamics.py` | discontinuous atlas, missing contact/state, tilt/travel in `test_independent_suspension_dynamics.py`; policy/force/refinement/state limits in `test_spatial_contact_dynamics.py` and `test_unilateral_spatial_contact.py` |

Both new Gym cases check observation-space membership, finite reward, active valid outcome, event ordering, no truncation/termination, one bounded landing followed by persistent grounded state, solver version, zero actuator work, nonnegative dissipation, every leaf energy residual/tolerance, and linear/angular impulse residuals <=1e-7. Original canonical world and fixture physics identities remain pinned after execution. Historical-selector failures exercise the actual public Gym entry and step, not a mocked kernel.

Independent numerical free-rotation comparison, passive-energy/momentum checks and version/hash/forged-audit tests remain in the existing suites. New rotating-version ridge/valley coverage supplements previous selector cases without changing their expectations.

## Evidence and remaining gates

Preceding source-pinned external records at8644db5 independently reproduce both50tick Gym fixtures; their checksums were reviewed. The new source tests preserve reproducibility in future exact-HEAD runs. Existing nominal four-track certificates and declared paired seed0/1/2,reset0 control witnesses are separate path/control evidence; they do not extend these recovery fixtures into arbitrary randomized robustness.

The inventory documents ramp_domain and mixed_normal_recovery as implemented following85 positive/negative/boundary/integration checks and independent Spec/Standards review. Final functional acceptance still requires source-bound verification and full exact-HEAD regression; this document does not claim those pending results. Weights, compatibility criteria and all unrelated capability states stay unchanged. Hosted Windows/Ubuntu CI and supported CUDA compatibility remain separate; local WSL does not satisfy independent reproduction. G1-G6 and training/heldout/scientific-freeze/WP8 permissions remain closed pending authoritative external evidence. No asymptotic settling, arbitrary mixed-normal terrain, empirical tire validity or scientific acceptance is claimed.
