# Road-relative suspension and bounded Ramp proposal

Base: Draft PR #29, `b24670dff81069979624eca1c4a036615757bd8b`.
Branch: `research/wp7-ramp-force-consistency-v1`.

## Outcome and remaining boundary

An explicit nominal engineering profile now drives the actual TRAIN `ramp-v1`
through its approach, incline, crest and upper platform, passes directed ordered
checkpoints/finish and conservative whole-body apertures, and brakes to a stop
after the full body clears finish. The recorded controls independently replay
through the existing isolated source/config/runtime-bound worker.

This is **not the registered PPO physics profile**. The proposal explicitly
selects `road_relative_vertical_v2`, a 0.2 rad numerical tilt domain and legacy
longitudinal Euler with zero steering. Existing registration retains the old
force model, 0.12 rad bound and distance ground integrator, which still rejects
incline contact. Proposal evidence records both registered and executed physics
hashes and `engineering_proposal_not_training_registration`. No registration,
owner contract, canonical assets, historical archives or readiness weights are
changed. Default planar audit still reports Ramp unresolved.

The model remains a road-driven reduced heave/pitch/roll system with prescribed
XZ/yaw motion. It is not a coupled tire/rigid-body force solver, physical ODE
validation, measured calibration, randomized robustness or research admission.
Incline drive/reaction work, traction/power limits, coupled inertia and momentum,
general steering and lifecycle transitions still require separate validation.
The zero-steering legacy longitudinal proposal must not be promoted to the
three-action research Gym contract, whose distance-integration guard remains.

## Reproduced defects

On a constant-velocity, road-following pose at z=80 m on the actual 8-degree
incline, world vertical speed is 0.4216225041 m/s. The old damper used that
absolute speed despite zero change in suspension deflection: at 2,000 N s/m it
removed about 843.245 N from each 2,943 N preload. A finite-angle force model
also needs the same gap geometry for generalized pitch/roll forces.

After fixing force consistency, a full nominal probe reached z=107.40758 m and
was falsely rejected at the crest. The upper-platform plane was extended behind
its finite patch: the rear of the car was 0.35 m below that infinite extension,
although it remained above the incline. The covered front corner was more than
0.20 m above the actual upper platform. Wheel support/travel remained valid.

## Versioned reduced force law

For each supported reference point p on a height field h(x,z), define the
vertical gap `g = p.y - h(p.x,p.z)` and compression `d = -g`. The opt-in model
uses `grad(g) = normal / normal.y` and the exact rigid-pose point velocity
`v_point = v + omega cross r`. The damper rate is
`g_dot = grad(g) dot v_point`, including road input from horizontal translation
and finite pitch/roll/yaw geometry. The vertical generalized force is
`F = max(0, preload + k*d - c*g_dot)`.

Pitch and roll generalized forces use the same virtual-work Jacobians:
`Q_angle = F * grad(g) dot (axis_angle cross r)`. The pitch axis is yaw-rotated
local X; the roll axis is yaw/pitch-rotated local Z. Constant reduced generalized
inertias and the existing deterministic semi-implicit substeps remain explicit.
They are not finite-angle spatial rigid-body inertia. Telemetry's existing
`vertical_wheel_velocity_mps` field is the damper input: legacy approximate
world rate in v1, exact gap rate in v2, identified by the policy force model.

The physical motivation is relative suspension velocity, as described in
[MathWorks' suspension-force model](https://www.mathworks.com/help/pdf_doc/vdynblks/vdynblks_ref.pdf).
The generalized-force construction is an engineering derivation using the
virtual-work framework described in
[MIT's multibody dynamics notes](https://underactuated.mit.edu/multibody.html).
Neither source supplies F1-RL calibration or approval of these parameters.

V2 requires exact angular-rate mapping and consistent stored chassis/suspension
rates. Simulation permits grounded zero-steering stepping only; partial support,
ballistic landing, manifold recovery and dynamic-yaw combinations explicitly
reject the new force policy. These guards prevent silently applying old forces
at a phase boundary. Legacy force behavior and its physics-document hashes are
preserved; new force/clearance identity uses physics document schema 3.

## Bounded-patch body clearance

`bounded_patch_planes_v2` clips each OBB face to the vertical prism above the
finite convex road polygon. Clearance is minimized over the clipped vertices.
Every vertex of this bounded intersection lies on an original OBB face: road
prism planes alone have rank at most two. Thus all six clipped faces supply
the vertices needed for the linear signed-plane objective. This is a bounded
halfspace-intersection construction; see the general formulation in
[CGAL's convex hull documentation](https://doc.cgal.org/6.1/Convex_hull_3/index.html).
No CGAL dependency is introduced.

For a swept interval the midpoint OBB's three half-extents are inflated by the
existing translation/rotation material-point bound. This encloses the defined
linear/SLERP substep motion before clipping. Coordinate-scaled numerical margins
expand the enclosure and patch prism and lower reported clearance. Nonfinite,
oversized, degenerate or unresolved intersections reject. This is a floating-
point numerical certificate with engineering tolerances, not exact arithmetic
or a bound against the continuous physical force ODE.

The historical infinite-plane verifier is still the default. The new reduced
force profile explicitly selects bounded-patch checking and records that choice
in its physics document. Existing wheel travel, road-union and obstacle checks
remain mandatory. A test with a narrow interior ridge rejects real body
penetration even when wheel travel and the body's distant corners are valid.

## Execution, replay and verification

`certify_ramp_execution(root)` and CLI `--ramp-proposal` opt in to the proposal.
The controller requests 3.5 m/s horizontal motion, accounts for the current
road's projected gravity and drag, and applies full braking after the centre
passes finish by 2.5 m. Commands are bounded by actual vehicle parameters.
All accepted suspension substeps pass contact/road/body checks; directed race,
hard 4 m corridor, full-body gate enclosures, speed cap and final stop remain
mandatory. No airborne states or steering commands are accepted by this profile.

The certificate has schema `wp7_executed_ramp_proposal_v1`. It retains the
existing source-origin, isolated-worker, runtime, full control tape and chained
trajectory bindings. Replay re-executes the tape and compares the entire result
against an independently retained certificate digest. It does not trust a
self-rehashed assertion. Actual profile physics is bound separately from the
unmodified registered protocol physics.

```powershell
python scripts/wp7_certify_planar_execution.py --ramp-proposal
python scripts/wp7_certify_planar_execution.py --replay ramp.json --expected-sha256 <independent-pin>
```

Focused evidence includes actual Ramp steady translation through Simulation,
independent geometric finite differences for gap rates and generalized torques,
legacy document hash pin, invalid mappings/rates, unsupported combinations,
crest false-rejection regression, interior ridge penetration and actual full
Ramp proposal execution/replay. Full regression, compile/dependency checks,
two-axis review and exact-HEAD Windows/Ubuntu CI are required before handoff.
The overall core goal and scientific gates remain incomplete.

Local full regression: **2,069 passed / 2 platform skips**. The 21 new tests
include 19 force/geometry/boundary cases and two actual Ramp execution/replay
and control-domain cases. Compileall, dependency compatibility (26 packages)
and `git diff --check` passed. Spec and Standards reviews have no remaining
material findings. Exact-HEAD Windows/Ubuntu CI results are recorded on the
stacked Draft PR before handoff.
