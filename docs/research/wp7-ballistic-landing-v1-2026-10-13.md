# WP7 3B — Ballistic airborne + guarded landing v1 (draft)

**Status:** research-only / NOT scientific acceptance; supersets PR #8. HELD-OUT remains SEALED. TrainingRunner's full-body training gate stays active; WP8 NOT STARTED.

## Implemented and independently testable contract

- Explicit `FlightState` phase: `grounded`, `takeoff`, `airborne`. No heuristic conversion from partial/missing wheel contact to takeoff: partial 1–3 wheel support is **NOT** solved by the current 4-wheel dynamics model.
- The authorized launch takes an already certified supported full-body/4-wheel pose with positive upward launch velocity and no rotation, preserves exact pose/velocity, changes `grounded` to false. Launch source/impulse is external and not scientifically registered yet.
- Exact gravity-only flight trajectory: `p(t)=p0+v0*t+(0,-g*t²/2,0)`; `v(t)=v0+(0,-g*t,0)`. No aerodynamic drag or rotational motion in this initial research domain.
- Full-body obstacle collision by swept OBB segments, with conservative world-Y obstacle inflation by `g*(h²)/8`, the maximum ballistic arc-to-chord error for each segment of length `h`. Only axis-aligned obstacle frames supported; rotated boxes are explicitly rejected (do not pretend there is no collision).
- Earliest eligible wheel/body collision with each horizontal planar patch solved by quadratic descending root, then checked for footprint overlap at contact time. Body-first ground collision is invalid; contact outside actual patch polygons does not count.
- Landing accepted only if downward touchdown speed <= **explicit engineering threshold**, landing tilt is within declared safety ceiling and precisely compatible with zero-pitch/roll grounded solver, all four wheels meet support travel, and the entire projected body/ground clearance is certified. No touchdown-speed number is asserted as an approved scientific criterion.
- A landing before the end of dt includes `remaining_dt_s`. The World integration consumes remaining time with the opt-in 3-DOF grounded suspension solver; never silently advances only a partial simulation tick. Flight onset and landing are reported as typed `FlightTransition` events with simulation-time offsets.
- While in takeoff/flight transition world steps, legacy point-volume checkpoint, finish and pickup triggers are suppressed. No extra reward or race finish is manufactured.
- Opt-in `step_world(..., independent_suspension_policy=..., ballistic_flight_policy=...)` with compatible contact policies and persistent `WorldState.flight`; unchanged legacy path when omitted.

## Limitations / BLOCKERS

1. **Automatic takeoff and loss-of-support detection** are not solved. User-authorized release is a stopgap, NOT proof that normal vehicle physics generated a jump. This PR must not be described as complete 3B acceptance.
2. No flight attitude integration / angular velocity; no landing tilt impulse or touchdown forces; airborne with nonzero angular velocity rejected.
3. Only horizontal road patches in a height-consistent 2.5D atlas; angled ramps and stacked decks rejected for touchdown even if the geometric path might be physically valid.
4. Exact ballistic arc collision is guaranteed conservatively only for world-axis-aligned OBB obstacles; rotated boxes yield an explicit error. No aerodynamic drag.
5. The `max_touchdown_vertical_speed_mps`, `max_touchdown_tilt_rad`, collision segment count and timing tolerance are research parameters, **not owner-approved scientific thresholds**. Inelastic impact sets vertical speed to zero; correct impulse/damping remains to be modeled and reviewed.
6. No complete continuous oriented-body vs sloped-road collision, no multi-contact impulse manifold, no closed-form force-aware transition, no reliable automated grounded-to-airborne release and no final scientific profile.
7. Flight-time progress/gate/reward semantics have not been versioned into an approved experiment. The new full-body PPO trainer remains blocked; HELD-OUT still sealed.
8. A flight start exactly touching road while descending is not a valid release. Future validation must explicitly cover grazing/zero-time contacts and ballistic arc crossing narrow elevated patches.

## Verification

Synthetic numerical, geometry and World integration tests only. Validate focused tests, full pytest, compileall, pip check on exact HEAD for Windows and Ubuntu through GitHub Actions; do not derive PASS from prior PRs.

**Do not merge into main/recovery, do not start WP8, and do not open HELD-OUT or freeze scientific protocol.**
