# WP3 Issue #51 — Three Frozen Replay Recovery, 2026-10-24

Branch: fix/wp3-eval-physical-completion
Worktree: D:/F1-RL-WP3-EVAL-PHYSICAL-COMPLETION
Source parent: ce0e55d7a428826b2d4ed3218641105bd0206f4b (uncommitted source edits)
Frozen TRAIN weights unchanged: 12 registered final model ZIP archives.
Historical roster: 96 cells, 5,760 slots, paired original model/environment/policy seeds.

## Same-build three required replay results

New scratch artifact: docs/research/WP3-V4AG-CURRENT-THREE-FROZEN-REPLAY-20261024.json
SHA-256: 7be0834a1f81fcd038969937065fda4d165ef0be77e1045baeac60e34631baff
Independent source checks: original model final ZIP SHA fields match provenance, individual paired episode environment and policy seeds match original frozen roster, and roster self-hash matches the original. 3/3 complete without runtime physics exceptions. All are explicitly NOT SCIENTIFICALLY CERTIFIED.

1. C-seed0 / right-turn-v1 / nominal / e0: ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED; failure_reason=physical_crash_development; success_flag_unaccepted=false.
2. A-seed1 / ramp-v1 / randomized_train / e17: ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED; success_flag_unaccepted=true; no crash terminal.
3. C-seed0 / slalom-v1 / nominal / e2: ENGINEERING_EPISODE_COMPLETED_NOT_SCIENTIFICALLY_CERTIFIED; failure_reason=physical_crash_development; success_flag_unaccepted=false.

The previously failing Road/Ramp cases no longer throw exceptions on these exact development replays, and the Slalom crash terminates at the development contract. This demonstrates technical execution, not a signed physics domain judgement. 90/90 targeted physics, crash, Gym, prefix reward, 12 model provenance/evaluator regression tests pass (exit 0).

## Evaluation status / explicit blocker

ALL_REQUIRED_REPLAYS_SCIENTIFICALLY_CERTIFIED=false; full_5760_eval_admission=BLOCKED in the replay artifact. Physics v4 has only development outcome version wp7_directed_lifecycle_crash_dev_v3; the crash validator explicitly rejects scientific_evidence=true. A fresh official experiment/evaluator/physics identity has not been registered/owner signed and source tree is dirty with uncommitted modifications. Therefore NO full 5,760-episode scientific Evaluation was launched. The original evidence remains untouched; no HELD-OUT, retraining, commit or push.

Before starting full evaluation, an authorized physics-domain owner must approve the source-v4 contact/release and RK4 event-time impulse contract as *physically-scored*, approve exact crash reward-prefix/evaluator semantics under a non-development registration and bind a clean source snapshot/sha, roster and seed policy to a fresh independent evaluation ID. The historical 5,760 slots cannot be overwritten or silently reused as valid proof. The WP2 full-suite tests that require a clean Git tree remain blocked in this uncommitted worktree, and broader regression failures have not been certified clear.
