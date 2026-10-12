# WP7 — Vehicle geometry review contract and registered TRAIN stochastic-action screen (2026-10-13)

## Decision: `review_required`. Scientific freeze and HELD-OUT remain BLOCKED.

All original PPO models, saved checkpoints, manifest/report hashes, tracks, physics, rewards, and source training configuration remain unchanged. This stage adds (1) an **unapproved geometry/corridor decision contract**, and (2) an evaluation-only **stochastic PPO action sensitivity screen** on the same four TRAIN tracks.

## 1. Explicit missing vehicle geometry

`configs/recovery/wp7-vehicle-geometry-review.yml` records the **historical** original vehicle wheelbase **3.0 m** and ride height **0.5 m**. The following remain `null` and **unapproved**:

- Body length, width, height, local body-centre offset, collision hull, tire contact footprint.
- Official allowable lateral deviation from authored route, body clearance beyond the union of road surfaces, and justification from course-design intent.

`f1rl/learning/geometry_review.py` is fail-closed: it rejects changed original wheelbase/ride height, unexpected schema fields, any filled-in unapproved dimensions, or fake approvals for training/HELD-OUT. Values from the earlier three swept-3D sensitivity profiles **are not vehicle specifications**. The `collision_decision` remains `undecided`.

Decision paths for future **separately reviewed** candidates: explicitly limit scientific claims to a *point-vehicle simulator* (without asserting physical body safety), or construct an actual body-aware continuous collision implementation in a **new candidate** with fresh training/evaluation. Neither is approved by this stage.

## 2. Predeclared TRAIN-only policy-action sensitivity protocol

- Uses the existing **two real final PPO ZIP archives** (seeds 0/0 and 1/1), with SHA256 checked and original TRAIN-source/manifest fingerprints verified.
- Same exactly four TRAIN assets: Straight Sprint, Slalom, Right Turn, Ramp. No new road, HELD-OUT access, physics/race/rule change, training, or weight update.
- Exactly **16 episodes per TRAIN track per model** = **128 real episodes**. The action distribution is sampled with `PPOModel.predict(..., deterministic=False)`; a separate deterministic seed is derived from SHA256(protocol identifier, archived run ID, TRAIN track ID, episode index), then `torch.manual_seed` is applied under isolated `fork_rng`.
- **Important limitation:** `F1RLEnv.reset(seed)` does not randomize this reconstruction's fixed spawn/road/track. This is **stochastic-action sensitivity**, **not environment generalization**, randomized vehicle geometry, different track, or an independent simulation-seed replicate.
- Predeclared screening threshold implemented before executing the evaluation: for *each* track, the **exact one-sided Clopper-Pearson 95% lower confidence bound of the success probability must be at least 0.80**. For 16/16 successes this lower bound is **0.82925028**; for 15/16 it is **0.73604260**; for 12/16 it is **0.51560358**.
- Binomial confidence intervals are conditional on the fixed model/environment and independent stochastic-action samples; they do not establish performance on unknown tracks. The 16-episode screening sample is small. The gate is never permitted to flip scientific freeze to `true`, even when a screening criterion passes.
- Results were observed **after** protocol values/seeds were fixed in implementation, but were not independently registered/approved externally before study execution; therefore this remains **exploratory**, not a preregistered confirmatory claim.

## 3. Real results

| TRAIN | Calibration PPO 0 / sampling 0 | Confirmation PPO 1 / sampling 1 |
|---|---:|---:|
| Straight Sprint | 16/16, LB95=0.82925 (screen PASS) | 16/16, LB95=0.82925 (PASS) |
| Slalom | 16/16, LB95=0.82925 (PASS) | 15/16, LB95=0.73604 (**FAIL**) |
| Right Turn | 12/16, LB95=0.51560 (**FAIL**) | 15/16, LB95=0.73604 (**FAIL**) |
| Ramp | 16/16, LB95=0.82925 (PASS) | 16/16, LB95=0.82925 (PASS) |
| Aggregate observed | **60/64** | **62/64** |

Both original final models previously completed 4/4 on **single deterministic TRAIN evaluation episodes**, and those earlier results remain correct. This action-sensitivity screen shows **Right Turn is a recurrent vulnerability under stochastic sampling**. Confirmation has fewer failures in this sample, but the evidence is insufficient to claim statistically superior policy quality.

Recorded failed episode seeds (not new training seeds):
- Calibration Right Turn: `3753800909`, `2502653887`, `1666344219`, `1636751858`.
- Confirmation Slalom: `4108866584`.
- Confirmation Right Turn: `1857273023`.

The overall predeclared screen **FAILS for both models**; no automatic tolerance adjustment or reward/route modification was made to force a PASS.

## 4. Evidence and safety gate

Immutable independent report:

`D:\Projects\.archives\F1-RL\wp7-exploratory\audits\wp7-train-stochastic-action-sensitivity-v1.json`

SHA256: `ce7b86c1869aa60766eb3565f343b967c5a76285cc90a22b355fc1dc367b196b`.

`scripts/wp7_screen_train_action_sensitivity.py` requires the two distinct archived seed roles and a new output filename outside original run directories or source Git. Its result contains 128 per-episode action RNG seeds, completions, rewards, steps and progress. Independent `f1rl/learning/robustness_screen.py` computes the exact binomial lower bound.

Freeze Readiness Gate **v3** additionally checks geometry review SHA256, stochastic-screen report SHA256, original model/manifest identity, all 128 registered seeds, success-count arithmetic, interval calculations, four TRAIN track identities, and explicit not-scientific/zero-HELD-OUT flags. Tampering must raise an error. It **still has six unresolved decisions**: official geometry, point-only physics, route/corner-cutting policy, missing historical provenance, observed stochastic sensitivity/absence of environment robustness, and verified independent backup.

## 5. Next scientific steps

1. Approve explicit vehicle-body specification and course-design route-corridor policy **without choosing thresholds to make previous policies pass**.
2. Resolve whether physical body collision is in scope. If yes, implement in a new versioned candidate, and re-train; never reinterpret legacy point-simulator scores as under the new physics.
3. Register prospective environment perturbation tests: varied spawn/heading/obstacle placement or wholly new TRAIN-only track scenarios with independent validation of validity. Keep HELD-OUT entirely sealed; do not use its scores to tune these protocols.
4. Record exact dependency, OS, Git, training config and model provenance *at next training time*, not retrospectively; independently back up archival weights/evidence and validate checksums.
5. Only after separately approved scientific freeze can a single registered HELD-OUT phase be considered.

**No PPO training, no HELD-OUT, no original artifact overwrite, no `main` merge, and no WP8 in this stage.**
