# WP3 Physics Validation Handoff — Khánh Băng

**Branch:** `fix/wp3-eval-physical-completion` (stacked on `fix/wp3-evaluation-physics-contract`, PR #53).
**Training source (unchanged):** `fe83f517479f3b990028d6409304fae83b8199a9`.
**Disposition:** **PARTIAL ENGINEERING FIX — NOT CLEARED FOR REGISTERED 5,760-EPISODE EVAL**.

## Implemented and testable

`f1rl/sandbox/multi_surface_support.py` adds a strictly opt-in `body_coverage_domain="wheel_supported_overhang_v1"` to `certify_multisurface_support`. Existing calls retain `full_footprint_v1` by default. The new mode is permitted only with `contact_domain="unilateral_free_extension_v1"` and `body_clearance_version="bounded_patch_planes_v2"`. It still verifies all four wheel support references and the whole continuous wheel sweep against the registered road atlas, travel/deflection limits, and bounded body/road nonpenetration wherever a surface is present; it does **not** mistake body overhang for absent wheel contact.

This is a low-level opt-in verifier, **not** a new fully registered simulator mode. It has not been enabled in WP3 evaluation or training, so it cannot change any published result. `tests/research/test_wp3_supported_overhang.py` covers the exact Right Turn seam geometry and swept yaw states, default strict rejection, explicit supported-overhang certification, missing support rejection, unknown mode rejection and invalid contact-domain rejection.

The 26 terminal-state mismatch instances were addressed earlier by PR #53, including real owner-failure integration tests, but the historical Eval remains `PASS_WITH_INVALID_CELLS` forever.

## Validation commands for Băng

On a clean checkout of this branch with the project's Python test environment:

```powershell
python -B -m pytest -q tests/research/test_wp3_supported_overhang.py tests/research/test_wp3_physics_domain_regression.py tests/sandbox/test_multi_surface_support.py
python -B -m pytest -q tests/sandbox/test_spatial_contact_dynamics.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_body_collision.py tests/sandbox/test_ballistic_flight.py tests/integration/test_spatial_compliant_gym.py tests/learning/evaluation/test_wp3_hard_route_terminal_eval.py tests/research/test_wp3_supported_overhang.py tests/research/test_wp3_physics_domain_regression.py tests/research/test_wp3_evaluation_issue51.py tests/research/test_wp3_eval_report_writers_issue51.py tests/research/test_wp3_campaign.py
git diff origin/fix/wp3-evaluation-physics-contract..HEAD --check
```

Before any model replay, verify the unchanged original 12 final-model SHA-256 manifests and sealed HELD-OUT. All scratch replays must use the frozen paired roster in `handoff/wp3-khanh-bang-20261021/evaluation/issue51-v1/evaluation/evaluation-roster.json`, a separate scratch output root and a newly recorded evaluation code digest.

## Open blockers — **do not claim resolved**

1. **Road-edge lifecycle:** support verifier no longer conflates wheel support and body footprint *when opt-in*, but the RK4 force solver in `spatial_contact_dynamics.stage` still requires all four contacts and throws `spatial wheel support/travel missing`. A scratch experiment ignoring the body footprint (77 occurrences) confirmed this later failure. Must implement verified partial-wheel force integration, phase brackets and continuation/terminal semantics with explicit approved `ground_motion_version` and physics source identity before any fresh full Eval.
2. **Airborne obstacle collision:** `world.py` detects definite SAT overlap and throws `FlightPhysicsError: no scored impact response registered`. The original `resolve_obstacle_block` rolls back a chassis velocity without momentum/energy audit, and `PhysicsConfig.validate_transition` rightly rejects that workaround. Need a separately registered first-contact-time and impulse/energy/audited absorbing terminal crash or physical impact response, with replay test from C-seed0/Slalom/nominal/episode 2. Do not suppress the exception.
3. **Canonical campaign state:** the original `attempt-v8` state is stale relative to 12 validated models, and historical preparation prerequisites were not all recovered. Mark BLOCKED rather than forge state or skip source/provenance checks.
4. **Official Eval:** historical 96-cell attempt remains 31 complete, 65 invalid. The new registered 96-cell/5,760-episode evaluation is NOT authorized until new physics, 3 frozen-seed failing-episode replays, source/seed alignment and all targeted regression gates pass. Any new Eval uses its own identity/root with both training and evaluation source commits.

**Owner boundaries:** do not retrain 12 models, do not change HELD-OUT, do not merge to `main`, do not alter training source `fe83f51`, do not rewrite or redraw the historical invalid cells. WP7 scientific acceptance and Gate 2 remain blocked.

This handoff is for **review and targeted validation of the opt-in subcomponent**, not authorization for Băng to run full Eval.
