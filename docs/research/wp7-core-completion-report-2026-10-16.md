# WP7 Core Completion Report — 2026-10-16

**Overall: incomplete.** Four engineering increments have been implemented and
reviewed, with additional Right-turn geometry, distance-ground-motion and executed planar feasibility increments, but seven subsystems remain below 90% and unresolved P0 capabilities
remain. Software regressions do not establish scientific acceptance. No weights
or thresholds were changed to improve these scores.

## Branches and review chain

Base: PR #22 `research/wp7-core-execution-closure-v1`,
`8aad106cd5d4ed72b640938669ce9098f07d552c`.

| Increment | Branch | Commit | Draft review |
|---|---|---|---|
| A continuous planar certificates | `research/wp7-core-completion-v1` | `81bad5315cc6ba95fb41cef94a050b001055bad1` | [#23](https://github.com/onsra520/F1-RL/pull/23) |
| B timed lifecycle race/reset | `research/wp7-lifecycle-race-closure-v1` | `c2768a3aa184b3862168eae092dc103f7567290d` | [#24](https://github.com/onsra520/F1-RL/pull/24) |
| C braking and isolated saved PPO inference | `research/wp7-control-evaluation-worker-v1` | `13189b76a3068e62f5d3c40b354d7362a8c2b817` | [#25](https://github.com/onsra520/F1-RL/pull/25) |
| D run manifests and trusted reviews | `research/wp7-run-evidence-review-v1` | `84ff1dbcd371f48ccccc92247d71ec213b8dcc0a`, documentation `e05b4cc7464da123600520dda0c8f5489f1be98b` | [#26](https://github.com/onsra520/F1-RL/pull/26) |
| E Right-turn arc geometry and corridor witnesses | `research/wp7-general-planar-path-v1` | `a14cecfac0183fa75beef37c3bae4a55db6f775c` | [#27](https://github.com/onsra520/F1-RL/pull/27) |
| F distance-consistent research ground motion | `research/wp7-distance-ground-motion-v1` | `42e1993f0f2821b34aa95f200483bc51cfc1fe35` | [#28](https://github.com/onsra520/F1-RL/pull/28) |
| G registered spawn and executed planar witnesses | `research/wp7-executable-planar-feasibility-v1` | `b24670dff81069979624eca1c4a036615757bd8b` | [#29](https://github.com/onsra520/F1-RL/pull/29) |
| H road-relative suspension and bounded Ramp proposal | `research/wp7-ramp-force-consistency-v1` | latest commit containing this report | stacked Draft PR on #29 |

No automatic merge. Canonical assets, historical archives and owner decisions
are unchanged. Owner-contract SHA-256 remains
`d9dd49cef664b083a30c98cf6b1d9205ca9a45522ed1ef431b49289edf72c0d1`.

## Verified functional matrix

Source: `configs/research/wp7-core-readiness-matrix-v1.json` with unchanged
100-point denominators. Fresh `evaluate_readiness(..., verify=True)` executed
the union of implemented capability tests: **162 passed**. These percentages
are the verified implemented weights, not coverage percentages or probability
of scientific success. Compatibility remains a separate unfinished gate.

| Subsystem | Functional % | Compatibility | Remaining P0/P1 | Evidence | Status |
|---|---:|---|---|---|---|
| Simulation runtime | 100 | IN_PROGRESS | None in matrix | `runtime/physics.py`, `runtime/simulation.py`, lifecycle/runtime tests | Functional PASS |
| Vehicle dynamics | 45 | IN_PROGRESS | P0 ramp domain; P1 mixed-normal recovery | `sandbox/manifold_flight.py`, `test_pv5_lifecycle.py` | Incomplete |
| Track feasibility | 45 | IN_PROGRESS | P0 all TRAIN feasible; speed trajectory | `research/wp7_track_feasibility.py`, planar certificates/tests | Incomplete |
| Race/episode | 75 | IN_PROGRESS | P0 owner-approved hard-route failure semantics | `learning/wp7_race.py`, `test_wp7_lifecycle_race.py` | Owner blocked |
| Gym/reset | 100 | IN_PROGRESS | None in matrix | `test_wp7_train_reset.py`, `test_wp7_lifecycle_race.py` | Functional PASS |
| Perception/control | 75 | IN_PROGRESS | P0 control feasible on all required tracks | `test_versioned_braking_control.py`, observation tests | Incomplete |
| Training/PPO | 75 | IN_PROGRESS | P0 independently admitted registered training | `wp7_execution.py`, binding/vector/runner tests | Admission blocked |
| Evaluation | 75 | IN_PROGRESS | P0 admitted stochastic PPO research worker | isolated saved-model fixture tests | Admission blocked |
| Provenance/admission | 75 | IN_PROGRESS | P0 independent approval and real restore | run manifest/signature/backup tests | External evidence blocked |

The new D implementation provides stronger engineering tools without claiming
the externally blocked final 25 points. The same applies to C's software worker.

## Four real TRAIN assets

Fresh alternative-path audit reloaded the registered TRAIN assets and verified
manifest/world/content identities. No HELD-OUT assets were opened.

| Track | Certified result | Unresolved physical requirements |
|---|---|---|
| `straight-sprint-v1` | Continuous planar body/steering reference plus nominal executed numerical physics/body/gate/speed/stop witness | Continuous physical ODE/tracking-error validation; randomized full-path execution |
| `slalom-v1` | C2 alternative geometry plus nominal executed numerical boost/braking/contact/body/gate/stop witness | Continuous physical ODE/tracking-error validation; randomized full-path execution |
| `right-turn-v1` | Three arc paths certify continuous body/road/obstacle/gate geometry | All three violate hard 4 m corridor; executable speed/contact/steering schedule remains unresolved |
| `ramp-v1` | Registered profile remains unsupported; separate nominal reduced-force engineering proposal executes ramp/crest/gates/stop and replay | Registered profile integration, coupled drive/reaction physics, lifecycle and randomized validation |

Straight/slalom reference curves and Right-turn certificates are sufficient geometric proofs within their
declared planar domains. The reference certificates are not physics rollouts;
G separately executes Straight/slalom in the registered numerical simulator. Right-turn's three exact
candidates have reproducible hard-corridor violation witnesses; this is not a
global infeasibility certificate. No asset correction is asserted or approved.
See `wp7-right-turn-arc-feasibility-2026-10-16.md`. Ramp's planar-candidate aperture failures do not prove the real ramp
impossible. See `wp7-core-completion-phase-a-2026-10-16.md`.

## Integrated execution and PPO infrastructure

Versioned lifecycle race/reset supports the verified bounded domains. Actual Gym
tests combine takeoff, airborne checkpoint passage, analytic event time, rebound,
landing/recovery and finish, including finish-versus-timeout precedence. Substep
ground motion and ballistic center laws are audited before publication. Reset
streams remain independent across seed/rank and Windows spawn. Historical C2
semantics remain separate. Owner `hard_route_violation_semantics` is still null;
uncertified route samples stay invalid, without invented failure reward.

The explicit three-value braking action preserves historical two-action models.
Saved-model space mismatches fail; no weight conversion is inferred. Ground
braking and airborne gravity-only control behavior are integrated through Gym.
`PolicyBoundTrainingConfig` extends the existing contract with actual vector,
race/action, observation/simulation and checkpoint-plan identities.

F closes a numerical steering exploit: legacy yaw used incoming speed while
translation used post-braking speed, allowing rotation with zero translation.
The explicitly registered `arc_distance_v1` integrates distance and yaw from
one force-frozen, stop/cap-aware bicycle law. Suspension substeps use that same
law at their time nodes. Actual wheel support preserves braking after the
centre leaves a platform; missing/ambiguous horizontal support is rejected.
Direct Gym and factory research braking both require this registration.
Historical Euler entry points/default configuration documents stay unchanged.
Current prepared TRAIN scenarios now bind the new schema-2 physics identity;
old receipts require explicit regeneration, without automatic model migration.
Ground sweep/event certificates still cover numerical substep chords/SLERP,
not an exact continuous physical force ODE. This does not close ramp support,
all-track speed schedules or physical calibration. See
`wp7-distance-ground-motion-2026-10-16.md`.

G fixes actual registered spawn preparation: road-authored spawn coordinates
were incorrectly used as chassis height, placing wheel references below travel.
An explicit versioned spawn reference now initializes chassis/suspension at
road plus ride height while preserving authored navigation and legacy equality.
All four real TRAIN factories reset and accept their first physics step.
The existing auditor can optionally execute nominal Straight/Slalom reference
control tapes and independently replay source/config/runtime-bound evidence in
a fresh exact-root worker. Numerical body/road/obstacle/contact checks, full-body
gate enclosures, ordered race/corridor checks and stopping all run on actual
registered assets. These are engineering witnesses, not learned policy or
randomized robustness evidence. Right-turn/ramp remain unresolved, so matrix
weights/statuses stay unchanged. See `wp7-executed-planar-feasibility-2026-10-16.md`.

The new isolated worker loads a real, saved **untrained** PPO model and executes
stochastic inference in software fixtures only. Model bytes and concrete factory/
reward identities are checked. Policy-action RNG is process-isolated and seeded
after loading, independently of episode environment RNG. Invalid episodes abort
without redraw. Exact per-cell CP95 remains minimum 60 trials and lower bound
>=0.90. Three paired seeds, two strata and the proposed 120 cells / >=7,200
episodes remain unchanged and unexecuted; cohort/multiplicity interpretation
requires independent review. No fixture result is admitted research evidence.

Registered runner attempts now capture preparation-time runtime/receipt identity
before model/artifact allocation. Artifact binding verifies existing file bytes;
it cannot infer training history or measured timestep provenance. Automatic
checkpoint/completion publication for future admitted runs remains backlog.

Ed25519 owner/reviewer signatures bind exact protocol/run manifests, purpose,
time windows and independently supplied trusted identities. Both successful
fixture signatures still leave every permission false. The actual trust registry
is empty. See `wp7-run-evidence-review-2026-10-16.md` for schema and CLI details.

H fixes road-input damping and finite-angle virtual-work forces in an explicit
reduced suspension model, plus a finite-patch body-clearance false rejection at
the Ramp crest. The actual TRAIN Ramp can execute nominal gates/contact/body/
corridor/speed/stop checks and exact replay in that engineering proposal. It
records both the registered physics hash and the actual proposal hash: the
0.2 rad numerical tilt domain and legacy longitudinal Euler are not silently
substituted into registered PPO scenarios. New-model steering, flight, partial
support and landing combinations reject. This remains road-driven reduced
dynamics, without coupled traction/drive-work or spatial-inertia validation;
it does not close the registered Ramp capability or change readiness scores.
See `wp7-ramp-force-consistency-2026-10-17.md`.

## Verification and compatibility

| Increment | Local full Windows | Exact-HEAD Windows + Ubuntu CI |
|---|---|---|
| A | 1,918 pass / 2 platform skips | 1,920 pass each; run `38857506598` |
| B | 1,936 pass / 2 platform skips | 1,938 pass each; run `38858890456` |
| C | 1,953 pass / 2 platform skips | 1,955 pass each; run `38861174631` |
| D | 1,975 pass / 2 platform skips; 22 new focused tests; 162 matrix tests | 1,977 pass each at `e05b4cc`; run `38862597646` |
| E | 1,989 pass / 2 platform skips; 38 focused geometry/readiness tests pass | 1,991 pass each at `a14cecf`; run `38903097375` |
| F | 2,023 pass / 2 platform skips; 112 focused integration/kernel/binding tests pass | 2,025 pass each at `42e1993`; run `38908204431` |
| G | 2,048 pass / 2 platform skips; 6 real-TRAIN spawn tests and 19 execution/replay/source-boundary tests | 2,050 full and 165 execution/readiness pass each at `b24670d`; run `38946178709` |
| H | 2,069 pass / 2 platform skips; 21 new force/geometry/boundary and actual Ramp proposal/replay cases | Exact-HEAD Windows/Ubuntu CI required; consult stacked Draft PR checks |

D compileall, dependency compatibility (26 packages), and `git diff --check`
passed before commit. Two-axis review has no remaining material P0/P1/P2 in
the D increment after fixing contradictory rehashed receipt acceptance. This
does not remove the unresolved subsystem P0 capabilities above.

Regression includes legacy C2 actions/observations/rewards, saved-model load,
Dummy/Subproc vector backends, seed determinism, evidence schemas and explicit
`prepare_scenarios(seed_index=...)`. No silent configuration migration is added.
The local runtime is Python 3.12.15, torch 2.14.1+cpu, NumPy 2.5.4, Gymnasium
1.4.0, SB3 2.9.0. CUDA hardware execution and other Python/dependency combinations
are unverified; CPU success cannot be promoted to universal compatibility.

## Scientific gates and minimal remaining work

| Gate | Status | Still required |
|---|---|---|
| G1 physics | BLOCKED | Ramp/mixed-normal/variable-omega contact domains and independently measured reference evidence |
| G2 calibration | BLOCKED | Authentic measurements, uncertainty and independent validation |
| G3 robustness | BLOCKED | Independent admission, registered research worker and real stochastic TRAIN cell outcomes |
| G4 reproducibility | BLOCKED | Authorized actual run-bound checkpoint publication and independent reproduction |
| G5 evidence recovery | BLOCKED | Actual private Google Drive archival and independently restored bytes |
| G6 scientific review | BLOCKED | Authenticated real trust bootstrap, independent reviewer and owner sign-off after G1–G5 |

Remaining engineering: general corridor-compliant Right-turn trajectory search; verified ramp/contact
support; executable speed/braking/boost schedules and tracking on all four tracks;
future admitted runner checkpoint observers and research-worker registration;
CUDA/remaining compatibility evidence. Remaining owner/external decisions:
hard-route semantics, measurement tolerances/calibration, cohort interpretation,
trust identity bootstrap, independent reproduction/Drive restore and sign-off.

`training_permitted`, `heldout_permitted`, `scientific_freeze_permitted` and
`wp8_permitted` all remain **false**. No new WP7 research PPO training, HELD-OUT
evaluation, scientific freeze or WP8 execution occurred. Historical exploratory
training remains historical exploratory evidence, not retrospective acceptance.
