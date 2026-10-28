# WP3 Issue #51 — Physics completion and evaluation gate (2026-10-24)

**Worktree:** D:/F1-RL-WP3-EVAL-PHYSICAL-COMPLETION
**Branch:** fix/wp3-eval-physical-completion (local, uncommitted)
**Handoff source:** ce0e55d7a428826b2d4ed3218641105bd0206f4b
**Original TRAIN source:** fe83f517479f3b990028d6409304fae83b8199a9
**Scientific status:** PHYSICS INCOMPLETE; FULL 5,760-EPISODE EVALUATION BLOCKED. No weights or old evidence changed.

## Implemented in this pass

- Road edge spring release ledger: f1rl/sandbox/wheel_edge_release_energy.py. Strict decreasing per-wheel support masks, positive stored potential release, <= 1e-8s bounded event, no rebound/energy creation; invoked by opt-in spatial v4 RK4. Energy is notionally dissipated into a road/environment heat sink, not calibrated. Recontact, ambiguous phase and unmatched balance fail closed.
- Geometry-to-impulse witness: f1rl/sandbox/certified_face_impact.py. Continuous first face-overlap bracket on an explicitly modelled linear+SLERP subsegment; unique obstacle normal and car hull point; calculate impulse, angular momentum change and dissipated kinetic energy using previously implemented rigid_body_impact_response.py. Ambiguous/grazing contacts reject. Hooked into opt-in v4 collision error diagnostic branch of f1rl/sandbox/world.py. No scored impact event yet.
- Hard wheel travel stop: f1rl/sandbox/wheel_bump_stop_impact.py. Bounded first crossed road-parallel bump travel, positive approaching normal impulse, angular impulse, before/after kinetic energy and restitution check. Opt-in v4 rejected-RK4-stage diagnostic now attempts it with exact rejected wheel/patch/state identity. This remains NOT_APPLIED and NOT_SCORED; accepted dynamics is unchanged.
- Evaluator: f1rl/learning/evaluation/evaluator.py now fail-closes any unregistered physical_crash_terminal input. Existing authorized terminal reason remains hard-route-violation only. New synthetic tests cover attempts to inject crash proof.

## Real frozen model/seed replays

New distinct full 3-case audit: docs/research/WP3-V4S-THREE-REPLAY-IMPULSE-AUDIT-20261024.json.
Original frozen 12 PPO weights and paired episode roster are read-only, original exact three model/track/seed records preserved. All 3 events remain INVALID, and the new registered evaluation count is zero.

1. C-seed0 / right-turn-v1 / nominal / e0 — SuspensionResolutionError: bounded body/road sweep witnesses physical overlap; first patch collision search already begins overlapping, so no certified first-clear-to-impact state/impulse. Not converted into route terminal.
2. A-seed1 / ramp-v1 / randomized_train / e17 — SuspensionResolutionError: front-right wheel deflection 0.120024443m exceeds fixed 0.12m. Extra scratch witness docs/research/WP3-V4R-RAMP-BUMP-IMPULSE-TRACE-20261024.json computes trial bracket 0.002709198 to 0.002709961s and prototype impulse Vec3(0,16.169448,-2.272468) N*s, nonzero angular impulse and energy loss 0.366792 J. This is only an interpolated *rejected RK4 trial*; applying it to accepted solver state is not yet implemented, so do NOT count as a valid terminal or physics repair.
3. C-seed0 / slalom-v1 / nominal / e2 — GroundedBodyCollisionUnscored: obstacle-3 rigid body collision. Scratch docs/research/WP3-V4O-SLALOM-FACE-IMPACT-AUDIT-20261024.json identifies first overlap bracket 0.000512695 to 0.000513458s, obstacle normal approx (0,0,-1), impulse approx Vec3(0,0,-2819.621) N*s, angular impulse approx Vec3(-2435.049,-185.979,0) N*m*s, and dissipated energy ~5489.414 J. Candidate is geometry/energy verified but NOT propagated into the simulation state or an approved absorbing scored crash terminal. Episode still invalid.

## Tests and admission

Combined handoff + source physics + evaluator regressions, including new edge release, wheel bump impact, face impact and forged-scoring denial: 259 passed, 0 failed, pytest exit code 0. Source used to replay still has no registered scientific owner-approved crash contract. This test count is NOT a model accuracy, reliability or physical-domain signoff.

Required next implementation: exact earliest first body/road contact and response, accepted RK4 event-time wheel hard-stop impulse and continuation/terminal, accepted source-registered collision terminal (with valid-prefix reward/event timing) for grounded/airborne contact, and independent physics/evaluator acceptance. Only when same three frozen episodes are VALID under the same newly registered source can an entirely new 96-cell/5,760-slot Eval be launched with immutable provenance and unique output root.

No re-training, HELD-OUT, commit/push, old Eval reuse/overwrite or scientific outcome promotion occurred.


## Latest source-validated full Eval release gate

Added f1rl/research/wp3_current_eval_gate.py and scripts/wp3_current_eval_gate.py. They read the original immutable model/seed roster and the latest three-sample physics replay, verify source base/roster SHA, final model ZIP SHA, exact paired environment/policy seeds, three case names and that the files are explicitly non-scientific scratch evidence. The gate returns BLOCKED_UNCERTIFIED_PHYSICS; its safe CLI returns exit code 3 on purpose. The 12-model immutable provenance preflight independently passed again (12 weights, 96 cells, 5,760 episode slots).

Latest comprehensive relevant test suite, including these release-gate tamper tests and all new physics/impact/evaluator tests: **273 passed, 0 failed**, pytest exit code 0. The actual three original frozen model/seed replay cases were run again into the distinct WP3-V4S-THREE-REPLAY-IMPULSE-AUDIT-20261024.json and ALL THREE STILL INVALID. No accuracy/finish-rate improvement or scorer approval is implied. The 5,760 registered Eval remains NOT RUN and must remain blocked until a newly accepted physical contact solver and crash-terminal evaluator are actually integrated and the real replay gate is passed.

## 2026-10-24 late source reconciliation — actual applied crash integration

Inspected the current worktree's extra opt-in applied transition implementation and ran the targeted tests. The current code includes f1rl/sandbox/accepted_contact_transition.py, accepted_road_body_contact.py, physical_crash.py, an opt-in world/simulation applied post-impulse crash, and a development-only owner-directed race/Gym/evaluator crash contract. tests/integration/test_wp3_applied_crash_end_to_end.py verifies applied linear/angular impulse in the actual Simulator VehicleState, an absorbing remainder, a tick-local impact timestamp, event prefix, and reward accrued only through the pre-impact state/time. 64 targeted tests passed (plus 9 tests for the latest source-bound V4V gate).

The latest three-frozen-case replay file is WP3-V4V-THREE-FROZEN-REPLAY-DEV-CRASH-20261024.json. Exact frozen model/seed verification: C-seed0 right-turn-v1 nominal e0 remains SuspensionResolutionError (first road contact starts already overlapping); A-seed1 ramp-v1 randomized_train e17 remains SuspensionResolutionError (hard-stop impulse only proved on a rejected RK4 trial, not accepted into its integrator); C-seed0 slalom-v1 nominal e2 completes a DEVELOPMENT-ONLY unsuccessful physical crash with failure_reason=physical_crash_development and scientific_evidence=false. Thus there are TWO invalid real physical samples, ONE development-only crash and ZERO scientifically approved samples. No full 5760 episode Eval has started.

New f1rl/research/wp3_issue51_latest_gate.py verifies original source roster SHA, all three frozen model SHA and paired original seeds from V4V and explicitly refuses to treat the dev-only crash as scientific completion; its 9 tamper/identity regression tests pass. This gate reports BLOCKED, 2 invalid, 1 development-only completion, 0 certified.

A broad research suite exposed two WP2 tests requiring a CLEAN Git worktree for candidate manifest generation. This isolated branch intentionally has uncommitted edits, so those WP2 tests fail by their original source cleanliness guard. Do not bypass the guard or claim a wholly green broad regression. The broader suite also recorded additional failing tests; inspect traceback before any claims of full source compatibility. No Git staging, commit, push or frozen training/evidence changes are authorized.
