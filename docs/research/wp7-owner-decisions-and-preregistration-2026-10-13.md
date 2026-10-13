# WP1–WP7 owner-approved target and scientific preregistration (2026-10-13)

**Authority:** user-approved technical choices 1B–2B–3B–4B–5A–6B–7B–8B.
**Scope:** implementation target for a NEW full-body vehicle/PPO candidate.
**Current decision:** DESIGN LOCKED. IMPLEMENTATION, SCIENTIFIC FREEZE AND TRAINING **BLOCKED**.
**WP0:** not a formal work package. **HELD-OUT:** sealed. **WP8:** not started.

## Owner decisions (immutable within this schema)

| Item | Owner selection | Interpretation |
|---|---|---|
| 1B | Balanced body 4.0 × 1.8 × 1.2 m | 3D chassis length × width × height; full-body collision, not point car |
| 2B | Simplified independent suspension | Four distinct wheel supports, independent travel; not detailed tire forces |
| 3B | Ballistic airborne + validated landing | Gravity/air motion plus explicit impact, road and collision checks |
| 4B | Soft route 2 m / hard 4 m | Horizontal authored-line distance; body must also respect road surface |
| 5A | Directed center crossing | Ordered checkpoint/finish center reference crosses oriented gate in forward direction |
| 6B | One-sided exact 95% CP lower bound >= 0.90 | Each TRAIN track, each independent policy, each predefined evaluation condition |
| 7B | Valid TRAIN spawn perturbations ±0.5 m / heading ±5° | Feasibility filtering and predeclared RNG seeds required |
| 8B | Private Google Drive | SHA256 evidence manifest + independent restore required; never public sharing |

The registered `configs/recovery/wp7-owner-decisions-preregistered-v1.yml` records these owner choices, deliberately marks implementation status pending and leaves all proof gates `pending`.

## Engineering reference values vs approved owner choices

- **Approved body envelope:** L4.0 × W1.8 × H1.2 m.
- **Provisional reference pose for feasibility review:** center offset [0,+0.3,0] relative to the original vehicle state position; original ride height 0.5 m; chassis bottom 0.2 m above an ideal flat road.
- **Provisional wheel points:** X±0.75 m, Z±1.5 m, Y−0.5 m; matches original wheelbase 3.0 m. Independent bump/droop travel of 0.12 m each is a *candidate engineering value*, not selected or validated against TRAIN geometry.
- **No selected safe touchdown vertical impact speed, landing collision response, suspension force law, road–wheel multi-patch assignment, route hard-violation recovery semantics or perturbation RNG protocol.** No code may invent those and claim scientific approval.
- The old `ppo-generalization-c2-v2.yml` maps **historical point-agent C2** behavior, not the future full-body model. Future physics/route semantics require a new experiment revision (potentially v3), separate policy identity and genuine retraining.

## Implemented preregistration controls

1. `f1rl/learning/owner_acceptance.py` strictly validates schema (including duplicate YAML key rejection), fixed owner decisions, four-wheel reference geometry, coherent ride height and wheelbase, explicit statistical/perturbation configuration, sealed HELD-OUT, no WP8 and no false approval flags.
2. `scripts/wp7_review_owner_acceptance.py` prints the review decision; `--require-ready` intentionally fails while any scientific acceptance remains blocked. It never trains or touches HELD-OUT.
3. `f1rl/learning/robustness_acceptance.py` implements **exact one-sided Clopper–Pearson**, keeping each independent PPO policy, each TRAIN track and nominal/randomized TRAIN condition distinct. The initial preregistered **60 episodes per track/model/condition** is a minimum budget; a raw 90% observed pass rate is not 90% lower confidence bound.
4. `f1rl/learning/corridor_acceptance.py` is a read-only continuous route screen. Distance-to-route is 1-Lipschitz in horizontal XZ space, allowing conservative midstep certification. Soft warning >2m, hard violation >4m, unresolved means invalid. **Not yet plugged into Gym reward/termination.**
5. `f1rl/sandbox/directed_center_crossing.py` detects forward-only crossing of a finite oriented gate by the vehicle's center segment. **Not yet plugged into ordered race-event resolver**; gate orientation needs future track acceptance. False legacy performance claims are prohibited.
6. CI tests all components on Windows and Ubuntu alongside existing 3D swept collision, flat road union and single-planar-ramp prototypes. CI pass validates code behavior against *current* assertions, not physical feasibility or statistical evidence.

## Training and release hard gates still outstanding

- **Vehicle:** approved wheel/tire dimensions, road-contact model, independent suspension travel, full-body supported multi-patch ramp, takeoff and landing including vertical impact validity.
- **Driving:** actual full-body lane union/road boundary and continuous authored-route 2/4m rules integrated with a versioned episode termination and reward contract; continuous ordered checkpoint/finish events integrated into race state; prevent shortcut or event ordering abuse.
- **Training:** generate a brand-new registered full-body canonical experiment, immutable TRAIN allowlist and source/track hashes, lock file, atomic **training-time** provenance, independent seeds (>=3 paired runs), valid TRAIN perturbation implementation, fixed statistical acceptance and artifact generation.
- **Backup:** private Google Drive destination created outside this public repository; model ZIP and evidence have **not** yet been independently uploaded, compared via SHA256 or restored. The GitHub repo must NEVER include private folder IDs, OAuth credentials or share links as source configuration.
- **Independent review:** evidence-backed signoff for all WP1–WP7 findings, re-run full cross-OS validation and verify historical PPO results on their own prior baseline. No retrospective relabeling of old weights.
- `TrainingRunner` currently deliberately refuses any prospective full-body mode before artifact creation. **Do not bypass this training block** merely because the owner config parses.

### Statistical interpretation

A 95% **one-sided** confidence lower bound >=0.90 is registered per independent policy, per TRAIN track, per nominal/randomized TRAIN condition. This is NOT by itself a family-wise 95% guarantee across all tested combinations. A multiplicity policy and pass/fail cohort definition must be independently decided and frozen before final validation. The empty final seed roster cannot be filled retroactively.

### Follow-up order

(1) multi-patch independent wheel support and proper ramp contact → (2) takeoff/landing → (3) event and route integration → (4) immutable run provenance + feasible randomized TRAIN resets → (5) new PPO training with >=3 independent seed pairs → (6) backup/restore/independent scientific audit. **HELD-OUT stays sealed until a separate post-freeze decision; no WP8 started.**
