# Simulation-only PPO recovery amendment v1

Status: **OWNER-SELECTED DESIGN; DRAFT — NOT PREREGISTERED — NOT AUTHORIZED.**

This amendment defines a distinct simulation-only engineering/demo track. Its scope is simulation consistency, source-to-runtime parameter identity and a bounded local Windows PPO demo. It makes no empirical calibration claim and does not satisfy original WP7 scientific acceptance. Machine-readable proposals are `configs/research/simulation-validation-v1.json` and `configs/research/simulation-only-ppo-v1.json`.

## Track A — simulation-only PPO recovery

The D1 fixture runs two seeded actual-environment replays on each of the four registered TRAIN tracks using a fixed action tape, reset identity, finite-state/space checks, transition audit checks and explicit absolute software replay tolerances. The protocol's coverage matrix links those fixtures to the existing deterministic physics, steering/throttle/braking, contact/suspension, flight/landing, route/finish, Gymnasium, observation/action and reward test suites. These tests establish software consistency within tested domains; they are not physical reference comparisons.

D2 binds all 12 WP7 measurement names and SI units to the selected runtime configuration where consumed. Absent measurements remain null; the active spatial traction coefficient is explicitly labelled an analogue, not tire calibration. The report carries configuration and runtime physics hashes and legal-domain source references. Sensitivity probes establish deterministic local parameter propagation and finite behavior only.

D3 proposes the owner-selected factorial: A=(penalty 2, corridor 2 m), B=(4,2 m), C=(2,3 m), D=(4,3 m), with three tuning seed blocks and 24 proposed confirmation blocks. The proposed budget is 65,536 learning steps/model, final checkpoint primary, 60 original-task episodes in each of four TRAIN tracks × two TRAIN strata, MDE .15, planning contrast SD .20, power target .80 and family alpha .05 across three Holm-adjusted contrasts. PPO seed blocks are the inferential units. The finite-sample power result and its independent-normal contrast assumption are documented in `D3-STATISTICAL-ACCEPTANCE.md`; neither the assumptions nor roster is approved as a statistical freeze. Secondary endpoints are enumerated in the machine-readable proposal. No evaluation outcomes were produced.

D4/D5 provide an explicit Windows-only `LOCAL_DEMO` command path, clean-source/config/track/observation/action preflight, fixed CPU and timestep/time bounds, telemetry/checkpoint evidence and non-learning dry-run. This path uses a process-local capability and is separate from scientific authorization; the owner direction removes Trusted Identity, signatures and replay ledger as prerequisites for LOCAL_DEMO only. The existing scientific admission/permit code remains isolated and operational. Demo artifacts are always classified non-scientific.

`straight_route_stability_demo_v1` remains fixed at 2 m; the separately versioned `straight_route_stability_demo_v2` accepts exactly 3 m for the C/D demo-training proposal. V2 is a local demo contract, not a change to scientific evaluation. Registered physics, observation/action contracts, original evaluation reward/termination, and the **4 m evaluation boundary** remain unchanged.

## Track B — original WP7 scientific acceptance

The existing `spatial_rotating_owner_v3` protocol and its physically calibrated acceptance remain separate and unchanged. Track B retains 15 seed-specific policies, 120 policy/track/stratum cells, 60 stochastic episodes/cell, final checkpoint as primary, per-cell one-sided 95% Clopper–Pearson lower bound ≥0.90, and abort-on-invalid/no-redraw. It still requires authentic physical measurements, independent physical comparison, owner/statistician decisions, scientific execution permit and authenticated trust, independent complete-run reproduction, authorized archival and restore, and owner/independent scientific review. HELD-OUT remains sealed.

The legacy `configs/research/straight-demo-abcd-v1.json` and `f1rl/research/wp2_statistics.py` remain proposal-only compatibility surfaces for their older 16,384-step/12-confirmation-block/sign-flip design. They are not inputs to this amendment or LOCAL_DEMO. They are retained to avoid silently rewriting historical run bindings; new Track A code reads only the versioned `simulation-only-ppo-v1.json` proposal. No source path upgrades either draft to authorized status.
