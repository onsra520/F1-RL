# Straight demo curriculum v2 proposal

Status: **DRAFT — NOT PREREGISTERED — NOT AUTHORIZED**. This document records a compatibility requirement only; it does not change the active validator, demo configuration, registered reward, or evaluation task.

The current `straight_route_stability_demo_v1` validator requires `training_corridor_m == 2.0`. Proposed WP2 cells C and D use a 3.0 m training stop corridor, so they cannot use v1 by changing the number. If C/D are approved, implement an explicitly versioned `straight_route_stability_demo_v2` contract that accepts exactly `training_corridor_m == 3.0` and preserves the existing bounded lateral penalty coefficient and terminal penalty rules. V1 must continue to accept exactly 2.0 m. Persist the selected version and corridor value in demo-training transition telemetry.

V2 may affect only demo-training reward and stop behavior after a valid transition. The registered 4.0 m evaluation hard-route boundary, physics, observations/actions, registered outcome checks, evaluation reward, and evaluation termination remain unchanged. Any implementation requires separate source review and owner/reviewer approval under Gate 2. Until then A/B/C/D remain DRAFT and no v2 runtime branch is enabled.
