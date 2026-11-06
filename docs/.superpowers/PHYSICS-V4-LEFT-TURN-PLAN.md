# Physics v4 PPO diagnosis, independent feasibility and TRAIN Left Turn plan

> For agentic workers: use superpowers:executing-plans task by task after the user approves this plan.

Status: DRAFT — awaiting approval. This document and its handoff evidence are preparation only.

**Goal:** Diagnose PPO on three existing TRAIN tracks, preserve the historical four-track contract, add a separately selected five-track TRAIN experiment, establish independent v4 turn feasibility, and smoke both turns.

**Spec:** docs/.superpowers/PHYSICS-V4-LEFT-TURN-SPEC.md. Read the handoff index and historical reports first.

**Inputs and outputs:** Use the root of the handoff checkout as REPO_ROOT. Historical input is under docs/.superpowers/handoff/evidence within the checkout and is read-only. Choose an OUTPUT_ROOT outside the checkout for new diagnostics, replay, test output and reports. Use an activated Python 3.12 environment and repository-relative paths; no fixed machine path is required. Historical scripts and reports may retain their original machine paths as provenance and must not be assumed portable.

## Global constraints and baseline

- Branch handoff/physics-v4-khanh-bang-20261106 must contain root baseline 2417570f32f6879c2651bce37d88fd6ac83a2ea4 as an ancestor. Handoff documentation commits may be later; verify that the source/config/assets/models match that baseline before implementation, accounting for only the approved handoff documents.
- Preserve four-track TRAIN_IDS, original four-track configs and manifest byte-for-byte, original TRAIN assets, frozen models and Phase 2A/2B/2C evidence.
- Do not touch HELD-OUT geometry, Physics/collision thresholds, PPO, reward, observation or terminal logic; no retraining or full 96-cell/5,760-episode evaluation.
- Keep output outside checkout. Do not commit, push or merge during execution of this draft plan.
- Fix a demonstrated blocker only within the permitted roster, harness, asset and test scope with a focused regression. If resolution requires a forbidden change or unavailable evidence, report BLOCKED with a reproducer and stop.

## Task 0 — Preflight and nine-episode PPO diagnosis

**Inputs:** Phase 2C report, phase2c-results.json and the nine Right Turn, Ramp and Slalom episode JSONL files, plus Phase 2A and 2B reports in the handoff evidence directory. The existing report already provides a preliminary diagnosis.

- [ ] Verify handoff evidence SHA256SUMS and baseline model/asset hashes; record exact Python/dependency versions, branch and source commit. Inspect tests and fixtures for HELD-OUT access before running them.
- [ ] For each of the 3 models x 3 tracks record finish, checkpoints, steps, reset/profile provenance, actions, position, speed, wheel-support fields and terminal witness. Provide direct file and step references; compare C-seed0's Ramp finish with the two Ramp crashes.
- [ ] Analyze turn entry and road position, Ramp approach and hard-stop witness, and the distinct Slalom trajectories. Identify missing measurements, competing policy/Physics/geometry explanations and confidence. Do not infer causation from a valid crash alone.
- [ ] If essential fields are missing, replay only the affected case using the same frozen model, v4 overlay, seed and step cap in an external runner; compare reset, outcome and steps before using new data. Record mismatches as blockers.
- [ ] Write OUTPUT_ROOT/driving-diagnosis.md and machine-readable case table, preserving all historical evidence.

## Task 1 — Freeze four-track behavior with regressions

**Preserve:** configs/experiments/ppo-generalization-v1.yml, configs/recovery/wp7-c2-candidate.yml, assets/tracks/manifest.json and the existing four-value TRAIN_IDS.

- [ ] Record byte hashes of all three historical files, four original TRAIN assets, 12 models and four-track reset registry. Record historical FULL-06 roster/cell counts and provenance from evidence.
- [ ] Add focused tests asserting original default scenario IDs, ordering, reset registry identity and FULL-06 4/96/5,760 behavior. Assert that an unrequested new roster is never used and inconsistent explicit candidate/manifest/roster selection fails before geometry or episode execution.
- [ ] Distinguish any source-bound fingerprint changes caused by source/test modifications from roster/reset changes; preserve the old result as historical evidence. Do not label newly generated source fingerprints as identical without proof.

## Task 2 — Add the separate five-track TRAIN configuration

**Create:** assets/tracks/training/left-turn-train-v1.json; assets/tracks/manifest-five-track.json; configs/experiments/ppo-generalization-five-track.yml; configs/recovery/wp7-c2-five-track.yml.

**Modify minimally:** the scenario-selection boundary in f1rl/research/wp7_protocol.py and only directly required callers/tests, without changing TRAIN_IDS or historical defaults.

- [ ] Create the new TRAIN asset by x-reflection of Right Turn; verify mirror geometry, ordered checkpoints, navigation, spawn, slow zone, finish and hashes using only TRAIN assets and metadata.
- [ ] Create a five-track manifest and experiment/recovery config separately, based on the frozen inputs, selecting the five ordered identities explicitly. Preserve the held-out identity in metadata without loading its geometry.
- [ ] Extend prepare_scenarios with optional explicit candidate_file, manifest_path and track_ids keyword arguments; omitted arguments preserve existing four-track behavior. A five-track call must provide a consistent registered candidate, manifest and ordered roster, with no empty, duplicate, unregistered or held-out IDs. Return the existing scenario interface.
- [ ] Exercise both paths with focused integration tests: the original four-track call produces the same roster/reset contract; the explicit five-track call produces the fifth scenario and its correct hashes. Confirm historical config/manifest/model/asset hashes and FULL-06 behavior remain intact.

**TRAIN-5 scope boundary:** Task 2 prepares an explicitly selected five-track scenario and engineering smoke only. It does not register or admit a complete five-track training protocol. The existing build_protocol() remains bound to four-track TRAIN_IDS and split_identity four_train_only_no_heldout_rollout. A separately scoped retraining phase must design and validate a versioned five-track protocol, split identity, roster/seed and cell definitions, admission and provenance before any official five-track training; retain the historical protocol unchanged.

## Task 3 — Independent feasibility of Left and Right

**Inputs:** TRAIN turn assets, registered v4 overlay and a deterministic non-PPO controller or control tape. **Output:** OUTPUT_ROOT/turn-feasibility/ with replayable action tape, reset and profile provenance, per-step traces, witness and report.

- [ ] P0 — Audit directed crossings before claiming either turn is feasible. For Right Turn checkpoint-0, checkpoint-1 and finish, record the gate orientation, local +Z forward axis, aperture and intended route tangent; repeat for the mirrored Left gates. The source currently has identity orientation for Right checkpoint-1 and finish while the exit road points +X. A mirror inherits this directional concern; do not infer that every possible path fails.
- [ ] Exercise both directed_center_gate_crossing and CenterSegment.directed_crossings with the real TRAIN gates. For Right checkpoint-1 and finish, verify that a center segment parallel to +X across the gate center produces no directed crossing and that an in-aperture segment crossing the gate plane toward +Z is evaluated according to the existing contract. Repeat the corresponding mirrored -X check for Left. Record inputs, output and source references; do not alter the crossing or race contract.
- [ ] In the independent v4 execution/replay, prove checkpoint-0, checkpoint-1 and finish are registered in order using actual directed crossing events and terminal telemetry, for each track whose feasibility is claimed. Geometric overlap, an arc certificate or entering the finish volume alone is insufficient. If ordered crossing cannot be established, mark that track's feasibility UNRESOLVED and the dependent causal/retraining decision BLOCKED.
- [ ] If resolving the crossing issue requires changing the original Right Turn TRAIN asset, stop and request review of a separate versioned-track proposal. Preserve the historical asset, manifest, config, models, thresholds and results; do not silently rotate gates, relabel historical results or relax the terminal rules.

- [ ] Check exact geometric reflection and construct/replay a reflected arc certificate. Report its geometric scope and unresolved speed/contact/steering assumptions; never equate this certificate alone with executable feasibility.
- [ ] Run a deterministic non-PPO control independently on each direction under the effective v4 physics/race/terminal profile, ordered checkpoint and finish rules. Use comparable nominal mirrored initial conditions if available; record actual reset and deviations. Limit the engineering search to five documented deterministic control configurations per direction, at most 2500 steps each; do not use PPO actions or alter Physics.
- [ ] Save every control input and terminal outcome; replay each successful tape with the same reset, profile and hashes. Require an actual valid Left finish with ordered checkpoints and no invalid transition for an existence witness. Run Right as a control and report its result separately.
- [ ] If no valid independent Left finish is found, mark feasibility UNRESOLVED and the causal/retraining gate BLOCKED; a bounded controller search failure is not proof the track is infeasible. Stop before PPO turn smoke and report the evidence. Do not loosen collision or finish rules.

## Task 4 — Physics v4 and historical compatibility verification

- [ ] Run the original 73-test command from REPO_ROOT using Python from the active environment: python -B -m pytest tests/research/test_wp3_smoke_runtime_profile.py tests/research/test_wp3_eval_only_coordinator.py tests/integration/test_wp3_registered_crash_terminal_contract.py tests/research/test_wp3_owner_train_dev_eval.py -q -p no:cacheprovider. All collected tests must pass; added tests may change the count.
- [ ] Run the new focused four/five-track tests and safe targeted v4 regressions for airborne road crash, wheel hard stop, road-crash prefix roundoff and registered terminal contract. Inspect fixture paths before execution; save exact commands, output and environment outside checkout.
- [ ] Verify requested/effective spatial_road_edge_partial_lifecycle_v4, wp7_directed_lifecycle_crash_contract_v4, wp3_physical_crash_terminal_v1 and max_abs_tilt_rad=0.7, mismatch fail-fast, valid crash telemetry, v3 compatibility and model/original asset integrity.
- [ ] Reconcile the eight Phase 2A v4 outcomes against Phase 2B and Phase 2C at episode level, and cite FULL-06 results as historical. Claim transition/substep equivalence only with direct evidence.
- [ ] Inspect tests/research/test_wp3_v4_technical_acceptance.py dependencies and all accessed paths. This test contains a machine-specific historical evidence path; the handoff archive does not automatically redirect it. Record missing data or unsafe HELD-OUT access as BLOCKED/NOT RUN, without fabricating evidence. State engineering coverage and scientific signoff separately.

## Task 5 — Paired deterministic PPO turn smoke

- [ ] Only after independent Left feasibility has a valid v4 execution witness and required in-scope blockers are resolved, run C-seed0, B-seed1 and A-seed2 on each turn: deterministic CPU, seed 20261106, cap 2500, no hybrid/auto-drive.
- [ ] For all six cases record actions, position, speed, checkpoints, reset, terminal, crash witness, effective profile, asset/model hashes and duration. Validate all crash witnesses and report invalid transitions.
- [ ] Compare Right with Phase 2C only to the extent reset telemetry and other inputs match. Mark Left as a zero-shot baseline; do not imply the models were trained on it. Distinguish episode-level and substep comparisons.

## Task 6 — Final report and per-direction retraining decision

- [ ] Save OUTPUT_ROOT/FINAL-REPORT.md, all traces, commands, hashes, independent feasibility witnesses and PASS/FAIL/BLOCKED/NOT RUN checklist. Separate finish from valid crash.
- [ ] For each turn direction independently, recommend a separately scoped retraining/optimization plan when all three frozen models fail to finish on that direction (0/3), its independent v4 feasibility witness is valid, relevant contract checks pass, no invalid transition occurs and relevant blockers are resolved. A successful opposite turn does not suppress this recommendation.
- [ ] If only some models fail, report the per-model results and diagnosis; recurring evidence may justify a targeted retraining proposal or more investigation. A single success does not prove retraining unnecessary. Six deterministic runs at one reset seed cannot establish statistical generalization.
- [ ] If the independent control, required Physics checks, identity/provenance, or crash-witness validation is missing/invalid, classify the associated decision BLOCKED or INCONCLUSIVE. Do not attribute the failure solely to PPO.
- [ ] If retraining is recommended, prepare a separate reviewable proposal with checkpoint-vs-scratch choice, roster, seeds, compute budget, allowable model/reward/observation changes, untouched frozen baseline, evaluation split, success/regression measures and stop rules. Do not start training under this plan.
- [ ] Stop and present the report. No commit, push or merge under this draft plan.
