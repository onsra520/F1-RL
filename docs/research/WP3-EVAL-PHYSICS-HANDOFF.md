# WP3 PPO Evaluation — Completion and Validation Handoff (Khánh Băng)

**Assigned:** @BANGPC123 (Khánh Băng) — GitHub Issue #51.
**Take over:** `fix/wp3-eval-physical-completion` (base Draft PR #54).
**Status on handoff:** Partial engineering groundwork delivered; **physics completion and fresh TRAIN Eval are assigned to Băng, not yet done**.
**DO NOT train again, alter HELD-OUT, or use the historical evaluation as if completed.**

## 1. Immutable inputs and branches

- Trained weights: **12/12** final checkpoints, each trained for **65,536 PPO timesteps**, 16 periodic checkpoints/model; original frozen training source `fe83f517479f3b990028d6409304fae83b8199a9` on `research/wp3-ppo-tuning-v2`.
- Training/checkpoint handoff: `handoff/wp3-khanh-bang-20261021`; see `CONTINUATION-20261022.md` and the private GitHub release `wp3-khanh-bang-20261021`, with the original three-run archive and continuation nine-run telemetry. Validate SHA-256 of both archives, all model ZIPs and telemetry before use.
- Existing historical evaluation: `handoff/wp3-khanh-bang-20261021/evaluation/issue51-v1`. **96/96 cells processed, 31 complete, 65 invalid-aborted; 2,153 valid episodes, 65 invalid episodes and 3,542 unattempted**. This result and its 204 evidence blobs must stay immutable. It is *not* a full valid 5,760-episode evaluation.
- Source stack: Draft PR #52 (`fix/wp3-eval-issue-51`, report-writer repair) → Draft PR #53 (`fix/wp3-evaluation-physics-contract`, terminal-failure repair/forensics) → Draft PR #54 (`fix/wp3-eval-physical-completion`, opt-in chassis overhang and first-contact bracket). Review merge dependency and provenance **without rewriting original training source or changing main prematurely**.

## 2. What is already implemented, and what it is NOT

1. `f1rl/learning/evaluation/evaluator.py`: when properly witnessed via `wp7_episode_outcome_v1` and owner hard-route metadata, `terminated=True/truncated=False` for `hard_route_violation` is a *valid failed driving episode*, not evaluator `ValueError`. This corrects the cause of **26 historically invalid cells**, but historical rows are unchanged.
2. `f1rl/sandbox/multi_surface_support.py`: explicit opt-in `body_coverage_domain="wheel_supported_overhang_v1"`, only with unilateral contact and bounded body clearance, certifies a chassis overhang *while four wheel contacts remain proven*. Legacy/default remains `full_footprint_v1`; no new simulation mode is registered or switched on automatically. Tests: `tests/research/test_wp3_supported_overhang.py`.
3. `f1rl/sandbox/body_impact.py`: `first_body_impact_bracket` returns a **bounded first-witnessed swept contact interval** with fail-closed uncertain/initially overlapping checks, including a thin-obstacle tunneling test and a synthetic Slalom geometry sample. It assumes **linear translation and shortest-arc SLERP**. It is **not yet** a certified ballistic-parabola first-impact solver; it does **not** implement collision impulse, restitution, work/energy audit, crash-terminal semantics or reward scoring. Tests: `tests/research/test_wp3_body_impact_bracket.py`.
4. Forensics: `docs/research/WP3-EVAL-ROAD-EDGE-IMPACT-BLOCKER.md` and `docs/research/WP3-EVAL-TERMINAL-PHYSICS-FORENSICS.md`.

The focused test group passed **70/70** and the expanded physics/evaluator/campaign test group passed **146/146** with the above source. These are **not** a full scientific acceptance or proof that the 39 remaining physics-invalid cells are repaired.

## 3. Băng's required completion work — finish BEFORE Full Eval

### A. Road-edge suspension / partial-wheel-release (34 invalid cells)

- Implement a **separately versioned opt-in** spatial contact/road-edge lifecycle. Current `spatial_contact_dynamics.stage()` requires all four `_contacts()` to exist and raises `spatial wheel support/travel missing`; the current four-wheel certification is not a partial-support integrator.
- Allow geometrically safe body overhang without assuming all wheels are permanently grounded. Compute contact/friction/suspension forces **only** for witnessed supported wheel(s); handle unload and release per wheel, with finite spring travel, support transitions and event-time brackets. When all wheels release, integrate physically valid airborne motion with linear/angular momentum, gravity and energy audits. Handle appropriate landing/recontact or an explicitly admitted absorbing terminal condition.
- Maintain full-body road **nonpenetration** and no false-clear/tunneling; fail closed on ambiguous contact, discontinuous road atlas, inconsistent support, energy/impulse errors.
- Register a **new physics/profile/version** in `PhysicsConfig` and factory/protocol wiring used by **new development TRAIN evaluation only**. Do NOT silently change the frozen `spatial_rotating_compliant_lifecycle_v3` contract.
- Regression requirements: all-four-supported, one-/two-/three-wheel support, total release, body overhang with supported wheels, road seam and edge, phase continuity, finite impulses/work/energy, and formerly invalid cases `C-seed0 / right-turn-v1 / nominal / episode 0`, `A-seed1 / ramp-v1 / randomized_train / episode 17`.

### B. Body/airborne obstacle collision response (5 invalid cells)

- Build from `first_body_impact_bracket` but certify **earliest swept first contact**, including ballistic/sagitta motion and rotation where appropriate; enforce explicit tolerance. Preserve fail-closed unknown/overlap states.
- Supply an opt-in, versioned **scored physical collision response**: appropriate normal, finite impulse and angular impulse and energy bounds, collision scoring, or explicitly registered **absorbing crash terminal at the certified impact time**. A crash can count as a **valid failed episode** only when supported by physical outcome and evaluator metadata; never as a finish.
- Do NOT call the legacy `resolve_obstacle_block` rollback then simply ignore `PhysicsConfig.validate_transition()`, and do NOT catch `FlightPhysicsError` just to mark an unmodeled physics transition valid.
- Tests must cover witnessed collision, glancing/slanted contact, high-speed tunneling, airborne parabola vs linear chord, initially overlapping body, uncertain contact, energy non-increase and audited impulse/time, and frozen reproduction `C-seed0 / slalom-v1 / nominal / episode 2`.

### C. Correctly isolated evaluation coordinator

- Original `scripts/wp3_campaign.py` only has **`run`/`resume`** and the authoritative campaign state may still be stale; it is **not** a standalone eval-only launcher. Do not call `resume` on the historical campaign if historical smoke/preflight artifacts are unavailable or state provenance does not match.
- Implement an **evaluation-only coordinator** if canonical campaign state remains unrecoverable. Read already verified **12 final models**, locked original paired roster and registered 4 m task as immutable inputs, and use a **new evaluation/source/physics protocol identity** and a **fresh distinct output directory**; record original training SHA plus evaluation source SHA and configuration digest. Do not forge original missing prerequisites or claim this is an authenticated resumed original campaign.
- Preserve fixed model/track/stratum/episode order: **12 × 4 × 2 × 60 = 96 cells / 5,760 episodes**; `straight-sprint-v1`, `slalom-v1`, `right-turn-v1`, `ramp-v1`; `nominal`, `randomized_train`. Paired frozen environment/policy seeds across all candidates; policy inference `deterministic=False`, CPU, 65,536-step final checkpoints only.
- Append-only episode logs and source/model/roster SHA-256. Invalid and interrupted attempts stay visible; **no silent retry/redraw/replacement of failed episodes**. New evaluation means a completely new run ID and evidence root, never filling invalid historical cells.

## 4. Validation sequence and exit gates

1. Branch checkout + confirm parent hashes and clean worktree, 12 trained final-model SHA-256, original evidence archives, registered evaluation roster. Record environment/CPU/library versions.
2. Run targeted sandbox and evaluator suite **before** replay:

```powershell
python -B -m pytest -q tests/research/test_wp3_supported_overhang.py tests/research/test_wp3_body_impact_bracket.py tests/research/test_wp3_physics_domain_regression.py tests/sandbox/test_spatial_contact_dynamics.py tests/sandbox/test_multi_surface_support.py tests/sandbox/test_body_collision.py tests/sandbox/test_ballistic_flight.py tests/integration/test_spatial_compliant_gym.py tests/learning/evaluation/test_wp3_hard_route_terminal_eval.py tests/research/test_wp3_evaluation_issue51.py tests/research/test_wp3_eval_report_writers_issue51.py tests/research/test_wp3_campaign.py
```

3. Add physically meaningful tests for partial support / collision response plus negative tests for invalid or unsupported trajectories. Run broad integration/full relevant pytest suite and a separate independent code review (not only tests written by the implementer).
4. In a scratch-only root (NOT the historical 96-cell output), replay the **three exact frozen model+seed cases** above. Require return of a certified valid *finish/failure* with no unscored physics exception. If any fails, **STOP: report blocker; do not launch 5,760 episodes**.
5. If (1)–(4) pass, register the **new TRAIN-development** Eval version; full 5,760 episodes with canonical 96-cell enumeration, raw per-episode events, final SHA-256 manifest. Require 96/96 cells with 60 valid episodes per cell to generate complete A/B/C/D descriptive ranking. If any invalid, preserve evidence and treat ranking incomplete.
6. Publish `WP3-TRAIN-EVALUATION.json`, `WP3-CANDIDATE-COMPARISON.md`, `WP3-STATISTICAL-READINESS.md`, `WP3-FINAL-STATUS.md`, 3 comparison/learning/failure SVGs and per-cell JSONL, source/config/model/telemetry SHA-256 evidence. Report finish rate, checkpoint progress, collision/hard-route, timeout, per-track and per-stratum, all 3 train seeds. Three independent PPO seeds are insufficient for confirmatory statistical superiority.
7. Push fix PR(s) and evaluation evidence separately; link each SHA/version to **Issue #51** for independent review. Leave Draft review gates until validated; **do not merge main without authorization**.

## 5. Scientific and release boundaries

**Training DONE, Eval and physics completion PENDING, scientific acceptance BLOCKED.** HELD-OUT must stay sealed, no WP8, no retraining all 12 models, no claim of scientific success based on engineering tests, no rewriting frozen source/weights or old evaluation files. If a required physics contract cannot be certified, document exact blocking cases and maintain failed safety gates rather than weakening them.

When finished: provide issue comment with feature branch + HEAD SHA, reproducible tests (counts/logs), all three previously invalid episode outcomes, full Eval summary counts, candidate comparison (if complete), ZIP/release/hash and PR links. Keep Issue #51 OPEN until independent acceptance.
