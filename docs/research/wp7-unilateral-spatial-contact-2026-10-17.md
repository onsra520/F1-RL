# Unilateral spatial spring/contact foundation

Parent: Draft PR #36, `669d20d49de3747363ec0890c384c3bed647f923`.
Branch: `research/wp7-unilateral-contact-v1`.

The opt-in `road_relative_unilateral_v3` keeps the existing spatial world-momentum integrator and height-field gap coordinate. For nominal preload P=mg/4, signed road-relative deflection d, stiffness k, gap rate v and registered damping c:

- E=max(0,P+k*d)
- L=E*max(0,1-(c/P)*v)
- spring potential U=E^2/(2*k)
- dissipation rate (E-L)*v >= 0

The force is L times the height-field gradient n/n.y. Unlike v2, it vanishes at zero elastic compression even for closing velocity. This is the clipped Hunt-Crossley form with exponent one in this generalized gap coordinate; see [Drake compliant contact](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html). The coefficient c/P matches the existing damping slope at nominal preload. This derivation does not constitute empirical calibration or identify a measured physical damping parameter.

At E>0 with an unclipped factor, dissipation is E*(c/P)*v^2. When the factor is clipped, v is positive and dissipation is E*v. At E=0, force, potential and dissipation are zero. Friction capacity follows normal load, so an unloaded reference cannot generate tire force. The existing RK4 energy, impulse and body/road sweep checks remain mandatory.

## Scope and evidence

One, two or three load-bearing springs are supported while all four geometric reference projections remain on the continuous 2.5D road atlas and within the existing suspension travel domain. Tests exercise these cases, an independent two-load force oracle, conservative zero-damping energy, momentum/work audits, and actual Simulation/Gym telemetry. All-unloaded steps still reject. Geometric partial release, takeoff, flight, landing and recovery are not implemented by this foundation.

The reduced suspension kernel rejects v3, rather than falling back to prescribed motion. PhysicsConfig requires the explicit spatial integrator; World preserves that restriction. The force-model string participates in the existing configuration hash. Existing v2 calculation, default registrations, canonical tracks, historical C2 contracts and scientific permissions are unchanged.

Focused regression: 56 passed, including nine new parameterized cases. Two independent read-only review axes found no material specification/physics or standards issues; neither reviewer ran tests. Full exact-HEAD Windows/Ubuntu regression and hosted results are recorded externally after commit. This opt-in foundation does not change readiness weights or capability status and grants no scientific permission.
