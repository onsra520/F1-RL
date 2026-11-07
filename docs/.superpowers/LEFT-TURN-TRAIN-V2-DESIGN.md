# Left Turn TRAIN v2 — independently authored geometry

**Status:** Engineering asset/schema design PASS. **No Physics v4 finish witness yet.** Training admission, PPO retraining and HELD-OUT access remain prohibited.

This design supersedes the *candidate exact-mirror TRAIN v1 only*, not the frozen four-track training experiment or the archived handoff. The exact reflected left-turn-train-v1 remains present for forensic comparison but is not eligible for TRAIN-5 because its world/content SHA-256 identifiers collide exactly with the registered historical HELD-OUT left-turn-v1 metadata.

## TRAIN-only authoring decision

The v2 route is newly parameterized from specified driving-geometry dimensions, **not copied or reflected from the historical Left Turn HELD-OUT file**, which has not been opened.

| Feature | Left Turn TRAIN v2 |
|---|---|
| Spawn | (x=3, y=0, z=5), facing +Z |
| Entry patch | Origin (3,0,44), heading +Z; width 16 m, length 88 m |
| Exit patch | Origin (-43,0,75), heading -X; width 16 m, length 92 m |
| Quarter-circle navigation reference | Radius 10 m, center (-7,65) in x/z; tangent +Z to -X |
| Checkpoint 0 | (3,0,38), gate forward +Z |
| Checkpoint 1 | (-24,0,75), gate forward -X |
| Finish | (-80,0,75), gate forward -X |
| Slow zone | z=46..64 m of entry patch, drag multiplier 1.8 |
| Vehicle body used for static area sampling | Width 1.8 m, length 4.0 m (existing research-body geometry) |

The new file is `assets/tracks/training/left-turn-train-v2.json`. It uses the existing authored navigation v2 schema and two planar patches. Gate orientation is authored from the outgoing route direction rather than silently reinterpreted by Physics or race code. No collision thresholds, reward, observation, PPO or terminal semantics were edited.

## Separate configuration (not historical replacement)

- `assets/tracks/manifest-five-track-v2.json` preserves the four original TRAIN rows and both original HELD-OUT metadata rows byte-for-byte at JSON field level and inserts one new fifth TRAIN entry.
- `configs/experiments/ppo-generalization-five-track-v2.yml` and `configs/recovery/wp7-c2-five-track-v2.yml` bind the fifth TRAIN identity explicitly while retaining the version-1 parsing schema. This is an *engineering scenario candidate*, **not** a registered TRAIN-5 learning protocol.
- The previous five-track-v1 files, original manifest and original recovery/experiment are unchanged. The current `wp7_protocol.prepare_scenarios()` opt-in selector still accepts only the v1 candidate; the v2 selector integration belongs to the later configuration-activation task, not this geometry-only task.

## Integrity and bounded validation

- Model archives, original TRAIN geometry/config and historical evidence remain immutable.
- v2 is checked against historical manifest metadata only, without reading any HELD-OUT geometry. Both world SHA-256 and content SHA-256 differ from every original registered TRAIN and HELD-OUT entry.
- `load_track_asset`, `load_track_manifest(validate_asset_ids=TRAIN-5)`, `RecoveryCandidate.from_yaml` and `build_training_scenarios` accept the new asset/configuration and construct five TRAIN factory objects without HELD-OUT geometry reads.
- Analytic static footprint: 303 reference poses, **35,451 sampled body points**, zero outside authored planar road union, minimum sampled boundary margin 3.0 m; this establishes **sampled geometric clearance only**, not continuous swept-volume proof or executable Physics.
- Both directed crossing implementations recognize intended +Z / -X checkpoint and finish crossings in static mathematical examples; actual order and terminal success must be proved separately with Physics v4 real replay.
- **158 tests passed** in the combined scoped regression (73 historical regression + 61 targeted Physics v4 + 24 focus including 6 v2 design tests), 0 FAIL / 0 ERROR, with JUnit external to the checkout.
- Raw artifacts and source-bound manifest/hash receipts: `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106\`.

## Outstanding gates

1. Finish separate **Right Turn v2** gate/geometry version design, retaining `right-turn-v1.json` unchanged.
2. Establish replayable deterministic **real Physics v4** feasibility, including ordered checkpoints and finish for both directions; do not infer from static geometry.
3. Add a versioned opt-in selector for v2 only after the complete paired scenario and provenance contract is reviewed. Design/register TRAIN-5 training seeds/splits and admission separately.
4. Do not train, fine-tune PPO, retune collision checks, consume HELD-OUT, overwrite any original model, or rerun FULL-06 under this design phase.
