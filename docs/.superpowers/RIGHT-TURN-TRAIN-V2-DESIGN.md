# Right Turn TRAIN v2 — versioned paired-direction design

**Status:** Engineering design and TRAIN-only scenario preparation PASS. **Actual Physics v4 ordered finish NOT YET VERIFIED.** PPO training and held-out evaluation remain closed.

## Authoring rationale

The historic `right-turn-v1.json` uses +X travel on its exit segment while checkpoint-1 and finish gate-local +Z are oriented along world +Z. The new `right-turn-train-v2.json` fixes that semantic mismatch in **a new asset**, not by editing v1 or relaxing the directed-crossing implementation.

Right Turn v2 is the x=0 reflection of the separately authored, independent TRAIN Left Turn v2; it is **not** a reflection of any HELD-OUT file. Mirrored v2 entry, corner, width and exit permit a controlled right-versus-left comparison. The historic four-track experiment is unchanged.

| Design parameter | Right Turn TRAIN v2 |
|---|---|
| Spawn | (-3,0,5), heading +Z |
| Entry road | Origin (-3,0,44), +Z direction; width 16 m, length 88 m |
| Exit road | Origin (43,0,75), +X direction; width 16 m, length 92 m |
| Turn reference | Rightward 90-degree quarter-circle, 10 m radius, center (x=7,z=65) |
| Ordered checkpoint-0 | (-3,0,38), gate local +Z -> world +Z |
| Ordered checkpoint-1 | (24,0,75), gate local +Z -> world +X |
| Finish | (80,0,75), gate local +Z -> world +X |
| Slow zone | Entry z=46..64, drag multiplier 1.8 |
| Vehicle dimensions used for static footprint sampling | 1.8 m wide x 4.0 m long, as in the existing research body |

The new exit checkpoint and finish gates use a +90-degree Y rotation of the original local-forward +Z. Both `directed_center_gate_crossing` and `CenterSegment.directed_crossings` accept forward +X geometric crossing and reject reverse crossing in the regression tests.

## New TRAIN-5 paired candidate (separate namespace)

- `assets/tracks/training/right-turn-train-v2.json` — new right-turn source file.
- `assets/tracks/manifest-five-track-paired-v2.json` — rosters [Straight v1, Slalom v1, Right Turn TRAIN v2, Ramp v1, Left Turn TRAIN v2], plus **unchanged** HELD-OUT metadata rows (manifest metadata only; no evaluation geometry opened).
- `configs/experiments/ppo-generalization-five-track-paired-v2.yml` — new draft/pilot-owned five-track experiment, not a frozen scientific training protocol.
- `configs/recovery/wp7-c2-five-track-paired-v2.yml` — separate C2 recovery candidate binding.
- `tests/research/test_right_turn_train_v2_design.py` — 8 test cases covering asset, symmetry, directed-gate crossings, hashes, manifest, factory and frozen preservation.

All original TRAIN-4 assets, original manifest, seed roster, 12 model files and archived evaluations remain untouched. Previously generated Left Turn v2 files/hashes and historical Left Turn v1 collision evidence remain intact.

## Acceptance scope

- Right Turn v2 asset passes the real project parser; both new v2 assets and three frozen original TRAIN tracks construct a five-factory candidate without reading HELD-OUT geometry.
- Both world SHA-256 and content SHA-256 are distinct from every original manifest row, and from their Left Turn v2 counterpart.
- Across 303 sampled poses and 35,451 body-footprint sample points, zero fall outside the union of the two authored road patches; minimum sampled road boundary margin is 3 m.
- **166/166 scoped test cases PASS** including Step1, 8 new Right Turn tests, old Phase2B/2C regressions, targeted Physics v4 regression. This is not a complete project-wide regression.
- These results validate **static geometry and contract structure only**. No actual dynamic v4 run, complete body-swept collision proof, wheel contact path, ordered checkpoint finish or PPO training has been performed for the new right turn.

## Phase boundaries

1. **Step 2: PASS** for versioned source, separate candidate files and bounded static validation.
2. **Step 3: NOT RUN**. Require independent reproducible Physics v4 control tapes and real finish for both turns, including correct reset, ordered gates, replay and collision/terminal semantics; unresolved otherwise.
3. **Step 4: NOT RUN**. The existing v1 opt-in selector is deliberately not silently retargeted to the v2 paired candidate. Register reviewed profile, seed/roster/approval source and actual candidate before any official TRAIN-5 run.
4. **PPO smoke / retraining / held-out / FULL-06: NOT RUN**. Preserve existing model provenance.

## Raw audit references

`D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\STEP2-RIGHT-TURN-TRAIN-V2-DESIGN-RECEIPT.json` and `STEP2-RIGHT-TURN-V2-STATIC-ROAD-CERTIFICATE.json`, plus combined JUnit XML/test log and follow-up acceptance report.

No commit, push, merge, historical result overwrite, or modification of other dirty worktrees is authorized in this design phase.
