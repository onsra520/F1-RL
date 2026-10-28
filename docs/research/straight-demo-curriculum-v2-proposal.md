# Straight demo curriculum v2 — implemented draft contract

Status: **implemented as a versioned local-demo reward contract; experiment remains DRAFT and unauthorized**.

`f1rl/exploratory/reward.py` supports two explicit demo-training contracts. `straight_route_stability_demo_v1` accepts exactly a 2.0 m training corridor; `straight_route_stability_demo_v2` accepts exactly 3.0 m. Both preserve the bounded lateral-penalty and terminal-penalty validation. `tests/research/test_demo_curriculum.py` covers accepted values, rejected cross-version corridor values, reward shaping and termination behavior. The separate `scripts/local_demo.py` runner records the selected version and corridor in training telemetry.

The proposed A/B/C/D configuration uses v1 for A/B and v2 for C/D. This makes the training proposals executable under the explicit demo-only contract, but does not authorize a run or change the experiment's DRAFT status. Any future run must use a fresh explicit local-demo command and an approved experiment scope. It must not claim WP7 scientific evidence.

The curriculum changes only demo-training reward and stop behavior after a valid transition. It does not change physics, observations/actions, registered outcome checks, evaluation rewards, evaluation termination, or the registered **4.0 m evaluation hard-route boundary**. A/B/C/D remain **DRAFT — NOT PREREGISTERED — NOT AUTHORIZED** until the required owner/statistical review and approvals are complete.
