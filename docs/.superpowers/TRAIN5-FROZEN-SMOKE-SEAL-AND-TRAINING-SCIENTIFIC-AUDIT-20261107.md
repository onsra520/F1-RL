# TRAIN-5 Physics v4 — frozen smoke seal & scientific training gate audit

Date: 2026-11-07. Owner request: complete (1) smoke SHA receipt, (2) training
contract scientific audit, (3) fail-closed regression & exact-commit CI, and
(4) train/evaluate **only if scientific gate PASS**.

**Branch:** `handoff/physics-v4-khanh-bang-20261106`

## Step 1 — six frozen PPO smoke cases sealed (PASS)

Original one-shot smoke source:
`01ef1e3e75de8df60fc2159ed70b38d0ee777253`

Executed six combinations of three existing, frozen PPO checkpoints and
Right/Left TRAIN v2, CPU deterministic prediction with reset seed 20261106.
Total original steps: **1,117**. Every case finished, traversed
`checkpoint-0` and `checkpoint-1`, and generated an immutable JSONL
action/reward/position/terminal stream.

A separate, read-only verifier checked all **6/6** original logs: SHA-256
against the original smoke summary; exact step sequence and count; finite
three-dimensional action vectors and rewards; identical source/model IDs;
two checkpoints and finishing records; source+origin Git identity; and all
**3/3 frozen model ZIP SHA-256** against the registered source and Git blob.

**No smoke replay was run during this verification.** No new checkpoint,
PPO training or HELD-OUT access occurred. GitHub CI
`42360810970` was SUCCESS on the source that produced the original logs.

External read-only receipts:

- `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\HANDOFF-01EF1E3-TRAIN5-PPO-SMOKE-LOCKED-ACCEPTANCE-20261107.json`
- Receipt SHA-256:
  `41d3ddf2a20801c442f2229ec241fb9aa1f7921f9ef518e575d1391362a58097`
- `HANDOFF-01EF1E3-TRAIN5-PPO-SMOKE-LOCKED-EVIDENCE-20261107.zip`
- Archive SHA-256:
  `d71a8c78aa38174947c4c9de69632e3755ed7db7cfdf21188971d1cea98eff2c`

This is **smoke validity, not a scientific benchmark or multi-seed
generalization evaluation**.

## Step 2 — exact training contract analyzed, but not admitted

Versioned, source-bound scientific audit:

`configs/research/wp7-train5-training-contract-scientific-audit-v1.json`

Five nominal TRAIN-only factories verified:
`straight-sprint-v1`, `slalom-v1`, `right-turn-train-v2`,
`ramp-v1`, `left-turn-train-v2`.

- **Action**: `steering-throttle-brake-v2`, three float32 values,
  with bounds [-1, 0, 0] to [1, 1, 1].
- **Observation**: 16 float32 features, 7 rangefinders plus 6 bounded
  signed features and 3 nonnegative features.
- **Reward terms**, equal across all five factories:
  `RouteProgressReward(weight_per_meter=1.0)`,
  `CollisionPenalty(penalty=5.0)`,
  `TimePenalty(penalty_per_second=0.1)`.
- **Physics**: `spatial_road_edge_partial_lifecycle_v4`; race
  `wp7_directed_lifecycle_crash_contract_v4`; physical crash terminal
  `wp3_physical_crash_terminal_v1`.
- **Seed roster**: 18 distinct pre-proposed integer seeds, allocated as
  3 components × 3 replicates × 2 planned cohorts. **No seed consumed**.
  This does not supersede the Global Seed Registry's explicit 15-log
  owner-accepted historical exception.
- **Proposed training**: three scratch and three frozen-initialized
  independent runs, each 200,000 PPO timesteps; 1.2M total. CPU, eight
  vector environments, 25,000-step checkpoint spacing; hyperparameters
  recorded in the audit. Frozen model A/B/C are **candidates, not
  automatically selected by smoke outcome**.
- **Proposed development**: 60 cells × 60 episodes = 3,600 TRAIN-only
  evaluation episodes. **Not executed.** No HELD-OUT, held-out
  selection or FULL-06 scientific evaluation.

### Scientific blockers, not waived by the user request

1. All five current TRAIN-5 v2 preview factories have
   **`reset_config=None`**. Nominal smoke with a single reset seed cannot
   validate independently sampled initial states, perturbations, or
   stochastic robustness.
2. Independent source-level scientific gates in
   `f1rl/research/wp7_train5_v2_protocol.py`,
   `TrainingRunner.run_multi`, and PPO backend
   `_validate_training_admission` deny research physics learning.
   **Do not modify these guards to manufacture PASS.**
3. No accepted empirical Physics v4 calibration/uncertainty envelope
   justifies a full experimental training distribution.
4. Reward/action/observation contract identities are now explicitly
   recorded, but the scientific development acceptance thresholds and
   checkpoint selection rules are not yet preregistered.
5. The owner approved attempting training, **not a fabricated
   declaration that these missing scientific prerequisites exist**.

Additional independent, read-only five-factory live introspection receipt:
\`D:\\F1-RL-PHYSICS-V4-EVIDENCE-20261106\\HANDOFF-01EF1E3-TRAIN5-LIVE-CONTRACT-NO-TRAINING-AUDIT-20261107.json\`,
SHA-256 \`8b9b3489c6dd66a2931745610ebb47445216a264f79e330747922ec475719772\`.
No episode was executed by this inspection.

Training review decision: **BLOCKED_NOT_ADMITTED**. Therefore the six
planned 200k-step policy training runs and the 3,600 development
evaluation episodes were **not started**; the source-registered 18 seeds
are untouched and the original frozen checkpoints unchanged.

## Step 3 — test-first and CI

`tests/research/test_wp7_train5_training_scientific_audit.py` locks the
source/track/seed/reward/action/observation contract and verifies
fail-closed authorization in both TrainingRunner and the PPO backend,
without model allocation or using training seeds.

The handoff's CI workflow includes these tests and independently
verifies the 44 pinned source blobs and frozen registration.

A separate exact-commit regression/CI receipt is required before
claiming Step 3 complete.

## Step 4 — conditional training and evaluation

**Blocked until all scientific gate requirements are satisfied.**
The next versioned, reviewable prerequisite should register an
evidence-backed TRAIN-only reset/perturbation distribution and empirical
validation scope, freeze numerical success/safety thresholds and
checkpoint-selection rules before development, implement an authenticated
training-specific scientific admission path with test-first regression,
and obtain separate final owner approval. Do not retune from
or access HELD-OUT. Training must run one-shot per approved seed with
source/checkpoint SHA capture and hard stop on unexpected errors.

No model weights, original history, sealed smoke logs or held-out assets
may be modified to bypass this scientific gate.
