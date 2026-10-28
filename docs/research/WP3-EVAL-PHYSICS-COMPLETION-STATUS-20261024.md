# Issue #51 — WP3 Physics Completion Gate, 2026-10-24

**Branch:** fix/wp3-eval-physical-completion
**Handoff commit:** ce0e55d7a428826b2d4ed3218641105bd0206f4b
**Windows worktree:** D:/F1-RL-WP3-EVAL-PHYSICAL-COMPLETION
**Issue:** https://github.com/onsra520/F1-RL/issues/51
**Source condition:** local worktree changes uncommitted; original frozen training SHA fe83f51 clean/untouched
**Outcome:** **FULL 5,760-episode EVALUATION IS BLOCKED — PHYSICS CONTRACT INCOMPLETE.**

## 1. Handoff verification

Fetched remote fix/wp3-eval-physical-completion at exactly ce0e55d. Original handoff states the source stack PR #52 -> PR #53 -> Draft PR #54, without authorization to forge missing physical outcomes. The supplied focused regression set passes **146/146** on its initial unmodified source.

The original dataset consists of 12 final frozen checkpoints (each 65,536 PPO training steps), 96 cells and 5,760 declared Eval episode positions, with source SHA fe83f51 and immutable historical evaluation with only 31 complete / 65 invalid-aborted cells. Do NOT train again, use HELD-OUT, or rewrite the original evaluation source/seed/weights/evidence.

## 2. Required three real frozen scratch replays executed

Script: scripts/wp3_physics_replay_preflight.py
Output: docs/research/WP3-CE0E55D-THREE-REPLAY-PREFLIGHT-20261024.json

Loads the *actual immutable original* model-final ZIPs (SHA verified against WP3-EVAL-SOURCE-PROVENANCE.json), source-locked paired evaluation-roster.json and the baseline rotating owner v3 TRAIN factory, CPU stochastic inference (deterministic=False), correct 4m route boundary and original per-episode seeds. NO TRAIN curriculum, no new training, no HELD-OUT.

| Frozen replay | Observed result | Full evaluation eligibility |
|---|---|---|
| C-seed0 / right-turn-v1 / nominal / episode 0 | SuspensionResolutionError | INVALID |
| A-seed1 / ramp-v1 / randomized_train / episode 17 | SuspensionResolutionError | INVALID |
| C-seed0 / slalom-v1 / nominal / episode 2 | FlightPhysicsError | INVALID |

3/3 cases still fail the physics certification gate. New physics semantics have NOT been implemented/registered. The tests above are real, read-only scratch replays and NOT historical evaluation restarts. Never count them as successful or completed Eval episodes. The replay report explicitly classifies them invalid and sets full_eval_gate=BLOCKED.

## 3. Safe source-grounded Eval-only preflight (independent of original campaign state)

New module: f1rl/research/wp3_eval_only_admission.py.
New CLI: scripts/wp3_eval_only_preflight.py.
Output: docs/research/WP3-EVAL-ONLY-COMPLETION-GATE-20261024.json.

Checks:
- original paired seed roster cryptographic self-hash, original provenance SHA, 96 unique (model, track, stratum) cells, 60 episodes each, 480 unique paired seed slots, canonical 4 TRAIN tracks and two strata, original 4m rule, stochastic policy/CPU final-only contract;
- 12/12 actual final model ZIP, original accepted TRAIN telemetry JSONL and model engineering-validation SHA-256, plus frozen state final-model digest; mismatch aborts before any new Eval directory could be created;
- original source commit and handoff source HEAD ce0e55d, original evidence file SHA-256, TRAIN track geometry hashes, exact three replay identities and exception classes;
- plan SHA-256, explicit BLOCKED gate and null new evaluation run identity. There is deliberately NO legacy campaign resume, missing smoke simulation fabrication or old evidence overwrite.

Result: 12/12 checkpoint/telemetry/validation hashes PASS; 96/96 roster cells structure PASS; three physics replays INVALID; **new registered evaluation count remains ZERO**. This is a standalone Eval-only **admission/preflight component**, not yet a full coordinator capable of executing 5,760 new episodes. A new coordinator/source/protocol registration still needs implementation once physical response and scored transitions are complete.

## 4. Independent ballistic contact diagnostic — partial physical progress, NOT impact response

New module: f1rl/sandbox/ballistic_body_impact.py.
New tests: tests/research/test_wp3_ballistic_body_impact.py.

This pure diagnostic locates a bounded earliest *witnessed* rigid-body contact on an exact constant-acceleration parabolic center trajectory, interpolating orientation by shortest-arc SLERP. Conservative midpoint displacement/rotation bounds attempt to certify all earlier intervals separated; uncertain contact or insufficient subdivision returns an explicit failure, never false clearance or a scored crash. Tests include a parabolic arc whose straight chord misses the obstacle, high-speed thin obstacle, initially overlapping bodies, and endpoint/parameter invalids.

The ballistic first-witness detector is NOT wired to the official simulator and cannot by itself create a valid crash or Eval result.

Added a separate pure rigid-body impact response diagnostic: f1rl/sandbox/rigid_body_impact_response.py. It takes an explicitly supplied, separately witnessed body-surface contact point, unit obstacle-to-vehicle normal, rigid mass/principal inertias and restitution; computes the world normal impulse, angular impulse, post-collision translational/angular velocity and kinetic-energy loss, and refuses separating, off-surface, nonunit or energy-generating inputs. **15/15 dedicated synthetic tests PASS**, including offset/glancing, high-speed, elastic and inelastic response and forged inputs. It is STILL not integrated: a body/obstacle manifold does not yet supply the certified contact normal, first ballistic TOI or owner-registered absorbing terminal. It does not authorize score/reward/valid episode and is NOT enough to permit full Eval.

## 5. Remaining physical implementation before Evaluation

A. Contact/road-edge: a new opt-in, separately versioned spatial contact integrator must carry 1/2/3-wheel support, bump/droop semantics, all-wheel release, ballistic flight, landing/recontact and energy/impulse audits. The source handed over still throws when RK4 stage lacks any wheel contact. It cannot be fixed by silently relaxing max_bump tolerance or pretending all-wheel support. Supported body overhang utility is already present but does not implement missing partial-support forces.

B. Ballistic/obstacle: the first-witnessed parabola diagnostic must be integrated with a versioned physical response (normal, finite linear+angular impact impulse, energy non-increase), or owner-certified absorbing crash terminal at exact certified contact time. Legacy rollback-then-ignore must not be used. Original FlightPhysicsError is still INVALID.

C. A truly new evaluation-only coordinator must be wired to the approved versioned physics and evaluator, with fresh unique output ID and root, appended per-episode JSONL and final SHA manifest. The current checker supports source admission only. NO full 5,760-slot evaluation should start until the above physics categories are certified and the *same three frozen cases* return valid terminal outcomes, with no unscored exception.

## 6. Testing and release status

- Handed-off target suite: 146/146 PASS before new edits.
- New ballistic collision tests: 11/11 PASS.
- New rigid-body normal/angular impulse and energy tests: 15/15 PASS.
- New Eval-only preflight tamper/roster tests: 16/16 PASS.
- Final combined original handoff + new physics primitives + v4 per-wheel tests + Eval-only preflight regression: **202/202 PASS**, exit code 0. Passing engineering tests do not override three real replay blockers.
- No new training, checkpoint mutation, official Eval, HELD-OUT, registered 5,760 episode run or scientific ranking.
- No commit or push yet. Branch fix/wp3-eval-physical-completion is based on ce0e55d with only new untracked diagnostic/physics primitive files. Issue #51 and Draft PR #54 remain open for partial-wheel 6-DOF integration, certified collision response and independent acceptance. Main is unchanged.

**This is incomplete implementation, not a claim of Issue #51 completion. Do not mark issue resolved or publish a successful evaluation.**

## 7. Additional opt-in v4 physics attempt — 2026-10-24

The draft physics variant **spatial_road_edge_partial_lifecycle_v4** is separately opt-in through PhysicsConfig, simulator world dispatch, F1RLEnvFactory and Gym. Legacy v3 is unchanged by default. No existing owner registration file is switched to this version.

New **spatial_partial_wheel_contact.py** witnesses each wheel independently as supported, beyond its authored road boundary, or above droop without conflating its cause. V4 integrates contact forces for only actually supported wheels. Synthetic regressions cover four, three, two, one and zero-wheel support; over-bump travel is rejected. A swept whole-chassis road nonpenetration check rejects unresolved geometry. This is a limited development prototype, **not** completed physics certification: the massless spring's stored energy may change discontinuously at a road edge, and the current RK4 audit fails closed at a critical edge event.

New independent scratch file:
docs/research/WP3-V4B-THREE-FROZEN-REPLAY-SCRATCH-20261024.json
Uses the original 12-checkpoint roster's exact three required model/seed combinations, CPU stochastic inference, original 4m task and opt-in v4. Not registered, not a new official evaluation.

Three actual v4 scratch outcomes:
1. C-seed0/right-turn-v1/nominal/episode0: SuspensionResolutionError — spatial energy balance unresolved at refinement limit.
2. A-seed1/ramp-v1/randomized_train/episode17: SuspensionResolutionError — spatial partial wheel penetrates road beyond bump travel.
3. C-seed0/slalom-v1/nominal/episode2: FlightPhysicsError — flight/substep body hit obstacle: no scored impact response registered.

An even earlier scratch file WP3-V4-THREE-FROZEN-REPLAY-SCRATCH-20261024.json recorded three ValueError due to a previously missing Gym opt-in version allowlist. That setup problem was corrected and recorded separately as V4B; it was not silently overwritten.

The following are NOT yet implemented/certified: physical spring release energy lifecycle at road edge, valid over-bump landing/contact impulse mechanics, a source-registered certified obstacle normal/contact point/TOI from the airborne body witness, integrated rigid-body impulse or owner-signed absorbing crash terminal, fresh executable Eval-only coordinator tied to the new identity, and independent acceptance.

**Actual mandatory frozen replay pass count remains ZERO of THREE. Full 5,760-episode registered Eval stays BLOCKED, no new score or candidate comparison.** No tolerance weakening, no historical evidence rewriting, no checkpoint retraining, no HELD-OUT.
