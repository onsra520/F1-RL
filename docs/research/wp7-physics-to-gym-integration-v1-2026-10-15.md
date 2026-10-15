# WP7 physics-to-Gym integration v1

Research branch: `research/wp7-physics-validation-v1`, based on
`362e47f1caec94dfbaa5ee27e1d6d03b6856e587` (Draft PR #18).
Software integration is not empirical physics validation or training admission.

## PV1: configuration handoff

Previously `step_world` accepted independent suspension, ballistic flight,
automatic takeoff and grounded yaw, but Simulation/Gym/factory only forwarded
the older body/ramp options. The runner only recognized the old body field.

`f1rl/runtime/physics.py::PhysicsConfig` now carries a versioned immutable
registration through `F1RLEnvFactory -> F1RLEnv -> Simulation -> step_world`.
Its canonical JSON/SHA-256 includes the complete nested policies (including
manifold, torque-free, CCD and analytic enclosure options). Gym reset/step
`info.physics` records that registration and solver telemetry, without changing
the policy observation/action spaces. Simulation snapshots retain suspension,
flight, partial-support and yaw audits from the actual world step.

Supported software research modes:

| Mode | Required registration | Limits |
|---|---|---|
| `legacy_v1` | No research policies/state | Historical fixed-step baseline |
| `wp7_suspension_v1` | Owner body, independent suspension, persistent state | Heave/pitch/roll; legacy-prescribed longitudinal/XZ/yaw |
| `wp7_ballistic_v1` | Suspension plus matching flight contact and flight phase | Existing guarded ballistic/landing domain; optional explicit edge release |
| `wp7_grounded_yaw_v1` | Suspension plus explicit matching yaw mass/axles | Planar dynamic yaw; low-speed domain rejected; no automatic takeoff |

No config preserves the historical API. Existing body/ramp prototypes remain
available through their original arguments, but cannot be mixed with a new
registration. Research state without an explicit mode is rejected instead of
silently ignoring suspension/flight. New registrations enforce the owner body
L4.0/W1.8/H1.2 m. Other engineering values in tests are synthetic assumptions,
not approved measurements. Tire hubs/patches are explicitly unsupported:
`contact_geometry` must name suspension reference points, not reinterpreted hubs.

`TrainingScenarioSet` compares the physics contract across tracks. Both runner
entrypoints reject the new research modes as well as the old body field,
before artifact/model creation. This strengthens, and does not replace or
remove, the existing unconditional full-body training prohibition. Legacy
training APIs are retained for compatibility; their availability is not owner
authorization to run a new research experiment.

Verification uses synthetic public seams: Simulation snapshots, factory-created
Gym reset/step, vector-slot info and runner admission. A displaced chassis
responds through suspension rather than legacy height snapping, yaw develops
from tire forces, ballistic position/velocity match a worked gravity case,
and low-speed rejection preserves the last accepted runtime snapshot.
Legacy observations/rewards/terminal flags/world telemetry are compared to an
explicit legacy registration. No new scientific rollout or model is produced.

## Scientific boundaries

True exact-ODE 6-DOF impact-time localization, near-contact recovery, measured
tire patches, unsupported mixed-patch impacts and empirical calibration remain
unresolved. An uncertain clearance interval is not an impact event. Existing
kernel exceptions propagate; no fallback or counterfeit success is introduced.

Owner decisions, historical C2 configs, model ZIPs and archived evidence are
unchanged. HELD-OUT remains sealed. `training_permitted`, `heldout_permitted`,
`scientific_freeze_permitted`, and `wp8_permitted` remain false.

## Reproducible software checks

```text
python -m pytest tests/integration/test_wp7_physics_integration.py tests/runtime tests/learning/training/test_fullbody_training_gate.py -q
python -m pytest -q
python -m compileall -q f1rl scripts
python -m pip check
python scripts/wp7_review_owner_acceptance.py
git diff --check
```

The regression suite includes its existing small synthetic PPO smoke fixtures;
these software checks are not preregistered TRAIN experiments or scientific
evidence. No exploratory runner, real TRAIN evaluation, HELD-OUT or archive
rewriting is part of this milestone.

PV1 local Windows Python 3.12.15 result: 1,774 passed, 2 platform-specific
skipped; compileall and `uv pip check --python <verification-python>` passed
(the reused verification venv has no pip module). Owner validator retained all
four false admission flags. Local script checks use `PYTHONPATH` set to this
worktree, avoiding the verification venv's older editable checkout.
