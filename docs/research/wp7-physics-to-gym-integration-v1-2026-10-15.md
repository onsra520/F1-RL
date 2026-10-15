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

## PV2: contact and recovery boundary

End-to-end Gym tests exposed a real handoff failure: World emitted
`FlightTransition` but both race progress and reward validation rejected it.
Both now accept that known world-event variant; it gives no race progress,
score or reward bonus. Synthetic four-point landings and restitution rebounds
reach Gym with event time, impulse/energy audit, and full-tick continuation.
The fixed-gravity contact time is independently checked by a quadratic root;
zero and 0.5 restitution cases check dissipative impact energy. These are
numerical/software scenarios, not measured vehicle validation.

Research-mode grounded obstacle contact now rejects the entire tick before
Simulation publishes state. The old rollback zeros chassis velocities while
retaining previous suspension rates, so it is not a coherent registered
physical recovery. Legacy rollback is unchanged. A grounded recovery law
needs explicit impulse/contact constraints and consistent suspension/yaw state
mapping; no such law is inferred here. Airborne suspension state retains its
last grounded value and is not incorrectly required to equal flight altitude.

Two coplanar patches support grounded four-wheel suspension. Simultaneous
mixed-patch landing remains rejected with unchanged snapshot. Torque-free
flight with nested RK4 CCD and analytic Euler clearance registration reaches
Simulation, but unresolved near-road enclosure still raises rather than
producing a landing/contact. No exact-ODE 6-DOF TOI solver is claimed.

PV2 verification: 184 focused tests passed; full Windows regression
1,780 passed, 2 platform-specific skipped. Historical baseline tests unchanged.

## PV3: grounded race boundary

`WP7RaceConfig(version="wp7_directed_grounded_v1")` explicitly opts into
directed center crossings over the actual accepted suspension substep path.
It requires registered grounded physics and authored navigation, refuses C2
offtrack mixing and ballistic mode, and records a SHA-256 over race settings,
navigation, checkpoint order/volumes and finish volume/orientation.

All continuous subsegments must pass the existing 2/4 m corridor verifier.
Soft >2 m is telemetry, without an invented reward coefficient. A >4 m or
unresolved sweep invalidates the episode; this is an engineering research
rejection rule, NOT an amendment to the owner's still-null scientific hard
violation recovery/termination policy. Gym requires reset after rejection.
Physical state remains owned by Simulation; a route-invalid but physically
accepted tick can advance Simulation internally, but no Gym sample is returned
and the episode must be reset. No rejected tick is recorded as a success.

Crossings use each authored gate's local +Z direction and finite XY aperture.
Events are processed chronologically, so multiple checkpoints and finish can
occur in one tick; an early future checkpoint is not replayed later. Simultaneous
different gates are rejected as ambiguous. Body overlap alone does not cross
the center gate. Physics rejection precedes all race/timeout decisions; valid
finish precedes timeout. Legacy C2 progress/reward/6m/10-step behavior is intact.

Ballistic directed race events remain unsupported: the curved flight path,
impact-time subsegments and gate ordering need a separate continuous contract.
Physics-only ballistic Gym tests remain supported without this race opt-in.
No reward bonus, soft penalty or new scientific admission is introduced.

PV3 verification: 92 focused tests passed; full Windows regression 1,789 passed,
2 platform-specific skipped, including historical Gym failure semantics.

## PV4: TRAIN reset preparation, admission still closed

`TrainResetConfig` accepts an explicit TRAIN-only manifest-entry registry and
independent uint32 seed. Evaluation/non-training paths, duplicate identities
and actual-world SHA-256 mismatches are rejected. Factory scenario IDs must
match the registered reset identity. The registry is a declared research input,
not an independent attestation that its contents are scientifically approved.

Opt-in Gym resets sample lateral offset uniformly in [-0.5,+0.5] m and heading
in [-5,+5] degrees from a SHA-256-derived stream keyed by protocol version,
perturbation seed, vector rank, track ID and reset index. Gym/SB3 reset seeds
do not rewind it. Multi-track wrapper recreation preserves the rank/reset
index; track selection uses its existing separate sampling RNG. No global
random state is modified. The metadata records exact draw seed, offsets,
attempt count, rejected draws, world hash and configuration hash.

V1 requires upright, stationary independent suspension plus the grounded race
contract. Each draw checks all-wheel/body road support, body/obstacle overlap,
route validity and gate-start feasibility. A bounded sequence of infeasible
draws fails reset; it never falls back to nominal or reports an invalid sample
as a valid episode. Nominal source state/navigation remain immutable. Alternate
Simulation reset states require explicit research physics and never replace
its constructor baseline. Spaces remain unchanged.

This research seed/filter implementation does NOT fill the null scientific
protocol fields in owner preregistration, approve a final seed roster or create
a runnable full-body experiment. The runner still refuses training before
artifacts are created. No real TRAIN/HELD-OUT rollout is run in this work.

PV4 focused software checks: 152 passed, 1 platform-specific skip, including
DummyVecEnv/SubprocVecEnv spawn, independent rank streams, redraw/no-fallback,
range endpoints, hash drift and runner admission. Full regression recorded
below after the final review. PV4 full Windows regression: 1,798 passed,
2 platform-specific skips.

## Scientific acceptance matrix

| Gate | Implemented | Software verified | Empirically validated | Status / required evidence |
|---|---|---|---|---|
| G1 physics | Bounded kernel-to-Gym handoff, guarded landings, directed grounded race | Synthetic gravity/impact energy, trajectory and failure contracts | No | BLOCKED: measured references, exact-ODE 6-DOF TOI/recovery and airborne race contract |
| G2 calibration | Existing measured-parameter/evidence schema | Units, completeness and hash contracts | No | BLOCKED: authentic measurements/uncertainty, independent physical comparison |
| G3 robustness | Existing numeric CP checks; feasible research reset implementation | Per-cell checker and deterministic reset/vector tests | No | BLOCKED: admitted versioned experiment, >=3 paired seeds, >=60 real episodes/cell, 95% one-sided lower bound >=0.90 |
| G4 reproducibility | Explicit nested physics/race/reset identities; existing run provenance | Hash/replay/config propagation tests | No independent replay | BLOCKED: training-time source/runtime capture and independent full-run reproduction |
| G5 private Drive | Existing manifests/offline byte comparison | Existing integrity regressions | No verified Drive recovery | BLOCKED: authorized private destination and independent off-device restore receipt |
| G6 review | Research decisions and limitations documented | Regression does not grant approval | No | BLOCKED: independent scientific review and explicit owner sign-off after G1-G5 |

## Scientific boundaries (unchanged)

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
