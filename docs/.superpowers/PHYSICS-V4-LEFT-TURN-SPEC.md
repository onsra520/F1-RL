# Physics v4 driving diagnosis and five-track TRAIN spec

Status: DRAFT — design option A is approved; this implementation plan has not been approved.

## Goals

1. Analyze the nine existing deterministic PPO episodes on Right Turn, Ramp and Slalom before changing the roster.
2. Add left-turn-train-v1 as a fifth official TRAIN asset for a new, explicitly selected five-track experiment. Mirror the existing TRAIN Right Turn about x=0.
3. Establish independent, replayable v4 feasibility evidence for both turn directions, then run paired PPO smoke and decide whether to propose a separately scoped retraining effort.

## Compatibility contract

- The original four-track experiment config, recovery config and track manifest remain byte-for-byte unchanged. The four identities and ordering in the global TRAIN_IDS remain unchanged. FULL-06 admission stays 4 tracks, 96 cells and 5,760 episodes.
- The five-track roster exists in a separate experiment config, recovery config and manifest, explicitly selected at the new scenario boundary. The fifth track is official TRAIN in that new experiment, and is not retroactively part of the historical experiment or frozen models' training provenance.
- This phase is TRAIN-5 preparation for engineering smoke, not a complete registered five-track training protocol. Existing build_protocol() remains tied to four-track TRAIN_IDS and split_identity four_train_only_no_heldout_rollout. Design and validate a separately versioned five-track training protocol, split identity, roster/seed and cell definitions, admission and provenance in the subsequent retraining phase before official training; preserve the historical protocol.

- Default callers keep their original four-track behavior. New five-track consumers must request the new candidate, manifest and roster explicitly; invalid or inconsistent combinations fail before an episode.
- Preserve historical reports and hashes. When comparing reconstructed results, distinguish stable four-track roster, assets and reset registry from source-bound protocol fingerprints. The source-bound fingerprint can legitimately change when new code or tests are added; report this instead of claiming identical source provenance.

## Geometry and feasibility

- left-turn-train-v1 reflects right-turn-v1 including road geometry, gates, navigation, spawn, slow zones and dimensions, while preserving ordering and distance. Its TRAIN manifest hashes must validate. Do not read or change HELD-OUT geometry left-turn-v1.
- P0 directed-gate concern: the Right Turn exit travels +X, while its checkpoint-1 and finish use identity orientation and directed crossing derives forward from gate-local +Z. Straight +X passage through the gate center is not a directed crossing under the existing functions. Mirroring the asset inherits the concern; this observation does not establish the cause of all PPO crashes or prove that every route is impossible.
- Before declaring feasibility for a turn, audit gate axes and crossing direction against the intended route, exercise both directed-crossing functions and verify an independent v4 replay records every checkpoint and finish in the required order. A geometric certificate or volume overlap alone is insufficient. If crossing cannot be established, report feasibility UNRESOLVED and block the dependent causal/retraining decision.
- If a correction requires changing the frozen Right Turn asset, stop and request approval of a separately versioned solution. Do not silently modify the original TRAIN-4 asset or relax crossing/terminal rules.

- A reflected arc certificate checks the declared geometric scope only. It does not establish speed, steering schedule, contact dynamics or executable feasibility.
- Independently of PPO, execute a deterministic control tape or controller on each turn with the effective Physics v4/race/terminal overlay, capturing replayable inputs, reset, provenance, ordered checkpoints and finish. A valid successful Left Turn run demonstrates existence for its recorded initial state and control only. If no independent control succeeds, classify feasibility as unresolved, not impossible; do not attribute PPO crashes solely to PPO.

## Evidence, decisions and limits

- Compare PPO behavior using the three frozen C-seed0, B-seed1 and A-seed2 models, deterministic CPU inference, reset seed 20261106 and 2500-step limit. Log every action, terminal and crash witness.
- A direction with 0/3 finishes can support a retraining proposal even when the opposite direction succeeds, provided that direction's independent v4 feasibility is established, relevant v4 checks pass and crashes are valid. Partial failures may also support a targeted proposal when the diagnosis provides evidence; state uncertainty and sample limitations.
- These results support an engineering decision; they do not establish statistical performance, scientific signoff, universal physics validity or that policy is the sole cause.
- Keep the physics engine, collision thresholds, PPO, reward, observation, terminal logic and frozen models unchanged during this phase. Do not train, run full evaluation, modify historical evidence, commit, push, merge or access HELD-OUT geometry during execution of this draft plan.
- Source changes belong only to the handoff branch. Store new diagnostic output outside the checkout. A separate user-approved scope is required to train.
