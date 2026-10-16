# Executed planar control-tape feasibility

Base: Draft PR #28, `42e1993f0f2821b34aa95f200483bc51cfc1fe35`.
Branch: `research/wp7-executable-planar-feasibility-v1`.

## Scope and result

The new auditor option `audit_train_tracks(root, execute_controls=True)` runs
nominal engineering control tapes on the actual registered TRAIN assets. Straight
and Slalom now have executable numerical-model witnesses: registered suspension,
distance-consistent drive/brake/steering, road/body/obstacle checking, continuous
body-aperture enclosures, directed ordered race progress, hard-4m corridor checks,
and a stop after complete body passage through the finish plane.

These are deterministic reference controllers, not learned PPO policies. No
research training, stochastic policy evaluation or HELD-OUT access is performed.
Only nominal spawn execution is certified; successful perturbation reset/first
steps do not establish full-trajectory randomized robustness.

Right-turn remains unresolved: the existing finite arc family certifies body/
road/gate geometry but violates the hard 4 m corridor. The executor does not
convert those paths to a PASS or claim global infeasibility. Ramp remains outside
the planar contact domain. No canonical asset or owner threshold changes.

## Real registered spawn defect

The first real execution exposed a preparation defect hidden by synthetic fixture
tests: authored assets store the spawn on the road at y=0, while registered wheel
points are 0.5 m below the chassis. Prepared factories had placed both chassis and
suspension at y=0. Actual reset rejected every perturbation, and nominal stepping
rejected wheel penetration before the first suspension substep.

`PhysicsConfig.spawn_reference="authored_horizontal_road_v1"` now explicitly
maps an authored horizontal road spawn to chassis position plus registered ride
height. `prepare_scenarios` applies that mapping and initializes suspension at
the same chassis height. Gym validates the mapping against actual road contact
and rejects arbitrary chassis heights. Authored navigation stays on the road;
its checkpoint/finish identities and coordinate system are unchanged.

Default `legacy_vehicle_position_v1` retains exact navigation/vehicle-start
equality. The default field is omitted from historical physics documents, so
their schema/hash identity is preserved. The new explicit reference appears in
schema-2 physics documents and their hashes. Protocol track rows additionally
bind initial WorldState hashes, with set-valued state fields canonically sorted.
Bound protocol/receipt identities must be regenerated; old evidence is not
retrospectively rewritten. All four real TRAIN factories are tested through
reset, observation validation and the first accepted physics step.

## Execution and continuous proof

The existing C2 quintic search supplies the reference. A deterministic bounded
curvature-feedback controller proposes steering and a 3 m/s speed schedule. Drive
and brake requests account for the actual current boost/drag modifiers. After
the centre is 2.5 m beyond finish, full braking proposes a stop. Every proposal
still goes through the actual registered `Simulation.step`; there is no teleport,
forced checkpoint, physics bypass or asset edit. Execution is bounded by the
registered episode step budget, at most 500 steps.

Registered physics checks wheel support, road coverage and obstacle sweeps.
`advance_directed_race` checks every numerical centre segment against the authored
route and directed checkpoint/finish order. The executor additionally encloses
every moving body between suspension nodes. At midpoint position/orientation,
inflation by half the segment translation plus `body_radius * full_rotation_angle`
(and numeric guard) encloses all body material points during shortest-arc SLERP,
including its tiny-angle fallback. If this box may intersect a gate plane, its
entire projection must fit the actual gate aperture. This is conservative and
can reject an otherwise valid passage; unresolved bounds never earn PASS.

The initial body must be wholly before the gates, the final body wholly beyond
them. All centre crossings must match authored order. A successful witness also
requires final speed <=1e-8 m/s and every recorded substep within the registered
speed cap. The terminal stop remains within the hard route corridor.

The reference curve has continuous bounded curvature. Executed geometry/contact
and race proofs cover the simulator's defined numerical chords/SLERP between
substeps. They do not prove a continuous-force physical ODE, steering-rate
regularity or a uniform tracking-error bound against the different reference
curve. Empirical calibration and general ramp/mixed-normal lifecycle remain
separate requirements; these witnesses do not close the all-TRAIN capability.

## Identity and replay

The public API rejects a foreign checkout root or a wrapper file changed since
import. Execution and replay run in a fresh isolated Python worker from that
root, with an empty temporary bytecode-cache prefix. Parent module objects and
monkeypatches are never used to simulate the tape. The worker checks imported
F1RL origins and source/config snapshots before and after execution; the caller
independently checks that source remained unchanged and validates the returned
digest. Source symlinks/escapes are rejected. A reproduced false-PASS case had
previously hashed an older checkout that did not even contain the new executor;
the wrong-root and parent-monkeypatch negative tests lock down that boundary.
These checks assume a trusted, stable host, not adversarial OS attestation.

The certificate binds actual asset world/content hashes, protocol/source identity,
physics/race/simulation/vehicle configuration, nominal initial state, proposal
settings, every steering/throttle/brake triple, final state, progress, speed and
substep counts, Python/platform/dependency identity and execution-source digest.
A chained trajectory digest covers each accepted WorldState,
physics audit and all suspension poses. Replay executes the recorded control tape
through the same validators and compares the entire regenerated certificate.
It requires an independently retained expected certificate digest; self-rehashing
modified controls cannot preserve successful evidence.

Source identity includes actual checkout bytes. Replay requires those exact
recorded source/config/runtime identities; CPU Windows/Linux test success alone is not
a claim of byte-identical evidence across different checkouts or dependencies.
The CLI emits JSON to stdout and does not overwrite evidence files:

```powershell
python scripts/wp7_certify_planar_execution.py --track straight-sprint-v1
python scripts/wp7_certify_planar_execution.py --track slalom-v1
python scripts/wp7_certify_planar_execution.py --replay certificate.json --expected-sha256 <independently-retained-digest>
```

An unresolved track returns exit 2, a verified executed witness exit 0. Admission
permissions are always false. Existing authored-only and alternative-geometry
audits keep their default API behavior; execution is explicit opt-in.

## Verification boundary

Tests cover both real successful TRAIN paths through the existing auditor,
control-tape replay, checkpoint/full-body/stop proof, actual four-track reset and
first step, invalid tape domains, rehashed false claims and intermediate rotational
aperture failure. Full regression, two-axis review, compile/dependency checks and
exact-HEAD Windows/Ubuntu CI are required before milestone handoff.

Continuation verification on 2026-10-17: full local regression completed with
**2,048 passed / 2 platform skips**. The new tests comprise six actual-TRAIN
spawn/reset/first-step cases and 19 execution/replay/source-boundary cases.
Compileall, dependency compatibility (26 packages), and whitespace checks passed.
Spec and Standards reviews reported no remaining material findings after the
foreign-root/source-pin defect was fixed. Exact-HEAD Windows/Ubuntu CI remains
required; its result is recorded on the stacked Draft PR.

The readiness denominator and capability statuses are unchanged. All-track
feasibility/control and speed certification remain blocked by Right-turn/ramp;
scientific gates G1-G6 and all four permissions remain closed.
