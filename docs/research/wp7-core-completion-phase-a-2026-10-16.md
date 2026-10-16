# WP7 core completion: first trajectory-certification increment

Parent: Draft PR #22, `8aad106cd5d4ed72b640938669ce9098f07d552c`.
This is executable geometry work within Phase 1, not completion of Phase 1,
physical validation, training admission or a >=90% readiness claim.

## Runtime entrypoint and certificate

```powershell
python scripts/wp7_audit_train_feasibility.py --search-alternatives
```

The default authored-path audit and its v1 output stay unchanged. Opt-in search
loads the same four hash-checked TRAIN assets and registered vehicle parameters.
It retains authored findings and adds independently hashed alternative-path
certificates; it never rewrites canonical assets or instantiates an RL policy.

Each planar segment uses `x(z)=x0+dx*(10u^3-15u^4+6u^5)` with increasing z.
The first two x derivatives vanish at segment joins, so heading and curvature
are continuous. The exact global bounds `|S'|<=15/8`, `|S''|<=10/sqrt(3)` give
`curvature <= |x''|`. This upper bound must fit `tan(max_steer)/wheelbase`.

For each interval of width h, bound center displacement from the midpoint by
`sqrt(1+max_slope^2)*h/2`, and yaw displacement by `max_second_derivative*h/2`.
Inflate the midpoint body's width and length by twice the sum of translation
and `body_motion_radius*angle`. The resulting OBB encloses the entire continuous
body trajectory, not just sampled poses. Existing road-union coverage and OBB
SAT prove enclosure clearance. Any non-clear interval rejects certification.

Whenever an enclosure can overlap a gate plane, require its entire tangential
projection to fit the actual rotated aperture. Then require strictly ordered
center crossings. Endpoint checks require the complete body before the first
gate and beyond the final gate. This conservative sufficient condition can
reject paths whose exact plane sections would fit; failure is not infeasibility.

## Real TRAIN findings

| Track | Alternative-path result | Still required |
|---|---|---|
| straight-sprint-v1 | Continuous planar body/steering certificate | Executed speed/contact schedule |
| slalom-v1 | Continuous C2 alternative passes all four directed apertures and body/obstacle/road checks | Executed speed/contact schedule, including boost behavior |
| right-turn-v1 | Finite graph-family search unresolved: steering bound and repeated gate-plane ordering | More general curve search; no global infeasibility claim |
| ramp-v1 | Unsupported noncoplanar/sloped support and unresolved elevated gate passage | Registered ramp transitions and contact dynamics |

Certificates serialize knots, curve version, interval resolution, body,
parameters, world hash, analytic curvature bound and all blockers. Replay is
deterministic. Source/config/manifest hashes remain in the enclosing auditor.

Speed/braking, tire-force feasibility, actual physics trajectory tracking,
initial-state compatibility and independent admission remain separate gates.
The readiness matrix's weights and unresolved statuses are unchanged.

## Verification scope

`tests/research/test_wp7_planar_trajectory.py` exercises real slalom and straight
assets, replay/hash stability, narrow road/aperture rejection, curvature limits,
complete finish passage, invalid/numerically degenerate inputs, unknown IDs and
honest right-turn/ramp blockers. Existing authored-path tests remain applicable.

Owner decisions, historical C2 behavior, archived evidence and all four false
scientific permissions remain unchanged. No research PPO or HELD-OUT is run.
