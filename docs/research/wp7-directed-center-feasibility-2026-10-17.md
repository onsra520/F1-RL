# Registered directed-center TRAIN feasibility

Parent: Draft PR #31, `11465189291ea47c5b6cc9ef7303e1b47ac018ae`.
Branch: `research/wp7-directed-center-feasibility-v1`.

## Contract and compatibility

Owner decision 5A requires: “Ordered checkpoint/finish center reference crosses
oriented gate in forward direction.” The registered YAML separately requires
full-body containment in the road union. Gate volumes are logical race triggers,
not physical obstacles. The prior optional whole-body gate-aperture certificate
is stricter than this owner contract; its failed candidates do not establish
Right-turn infeasibility under 5A. No authored gate orientation, canonical asset,
owner threshold or permission changes here.

`certify_arc_trajectory(..., gate_contract='directed_center_crossing_v1')`
explicitly selects schema `wp7_arc_path_certificate_v2`. Continuous full-body
road and obstacle enclosures, bounded curvature, actual finite gate aperture,
forward crossing and gate order remain checked. Passage records claim center
crossing, not whole-body aperture clearance. The existing default remains
`whole_body_plane_section_v1`, with the prior schema and stricter behavior.

The Right-turn reference is a C1 sequence of lines/circular arcs with bounded
piecewise-constant curvature, including curvature jumps. Its finite candidate
family uses curvature at most .16/m, retaining steering reserve for numerical
tracking, and the existing authored assets. This is a nominal witness, not a
global path-search completeness or continuous steering-rate proof.

## Actual registered execution

`certify_registered_center_execution()` reuses the fresh exact-root worker,
registered factories and numerical physics. Line/circle projection supplies
the Right-turn controller's reference and bounded steering feedback. Existing
speed feedback, modifier handling and explicit braking execute through
Simulation. The actual ordered directed race/corridor checks and all numerical
body/road/obstacle/contact checks remain active. The car must finish and stop.

The new `wp7_executed_registered_center_feasibility_v1` document includes the
explicit gate contract, owner-file hash, physics identity, source/config/runtime
fingerprints, trajectory digest and full control tape. It omits the old
whole-body gate-passages claim. Replay dispatches this schema to the same worker
and requires an independently retained checksum plus exact regenerated record
equality, rejecting even self-rehashed contradictory gate claims.

`audit_train_tracks(..., registered_center_gates=True, search_alternatives=True,
execute_controls=True)` reports schema `wp7_body_track_audit_v2`. Its authored
path diagnostics retain their historical scope; alternative and executed rows
carry their own contracts. Straight, Slalom and Right-turn have nominal executed
center-gate witnesses. Registered Ramp remains unresolved. Separate reduced and
force-coupled Ramp proposals are not silently substituted into registration.

CLI entry points:

```text
python scripts/wp7_certify_planar_execution.py --registered-center-track right-turn-v1
python scripts/wp7_certify_planar_execution.py --replay CERTIFICATE.json --expected-sha256 RETAINED_PIN
python scripts/wp7_audit_train_feasibility.py --registered-center-gates --execute-controls
```

## Verification and remaining scope

Public-seam tests cover actual Right-turn geometry, the unchanged strict-mode
rejection of the same path, actual four-TRAIN audit, Simulation-to-Gym completion
with valid observations, braking/stop, exact worker replay, rehashed claim
rejection, finite aperture/order failures, narrow-road/body-side obstacle
rejection and invalid contract selections. Windows/Ubuntu CI includes this suite.
Exact-HEAD regression results and independent CLI certificates are recorded in
the Draft PR and external milestone receipt after verification.

Numerical collision/race certificates enclose the defined substep chords and
SLERP, not a global physical-force ODE or physical tracking-error bound. The
controller is scripted; this is not PPO learning or stochastic robustness.
Randomized complete paths, physical calibration, registered Ramp, coupled
steering/lifecycle, owner hard-route semantics, independent admission and actual
Drive restore remain open. Readiness weights/statuses and all four scientific
permissions remain unchanged and false.
