# WP7 — Retrospective freeze readiness decision (2026-10-13)

## Verdict: `review_required` — **NO scientific freeze; HELD-OUT remains sealed**

This is a **post-training evidence verification**, not approval of model transfer or generalization. No PPO training, HELD-OUT evaluation, physics change, or source-model mutation is performed by this stage.

Registered original exploratory runs:

| Role | PPO/sampling seed | Actual steps | Archived TRAIN | First 4/4 checkpoint |
|---|---|---:|---:|---:|
| Calibration | 0/0 | 200,704 | 4/4 | 175k |
| Confirmation | 1/1 | 200,704 | 4/4 | 50k |

Each result is **one deterministic evaluation episode per TRAIN track per milestone**. Eight original report hashes and seven checkpoints plus final model were verified per run. Both runs' original manifest source fingerprints match the registered eight candidate/profile/base-experiment/TRAIN-asset files. The original run source commit is `1478cf6b9f671a08c5ace3878c8b19f34da951cc`.

The original `state.json` files **both contain `"provenance_file": null`**. This report intentionally **does not fabricate missing run-time OS/library versions**; it links existing model hashes, 200k hyperparameters, source revision, and subsequent audit hashes. Inspection-time package versions, if collected, must not be mislabeled as training-time.

## Independent evidence bundles

- Original four-TRAIN-only point-to-obstacle + route deviation audit: `dual-seed-train-trajectory-audit-v1.json`, SHA256 `0e44bcaf738f5c4eadb2f726b3a8108af64d1690f7a9fbf28ccbdabcd771d188`.
- Original 3D hypothetical-swept-body audit: `dual-seed-swept-3d-body-proxy-v1.json`, SHA256 `da1accdd0d0f5bdafb7e790466f8c30a6a054eb637b9c1dc6820b2114e2bfed8`.
- Both linked audits must contain the same **exact two run IDs**, model SHA256 hashes, original manifest SHA256 hashes, source commit, four TRAIN track IDs, and zero HELD-OUT rollouts. Tampering or source drift raises an exception, not a warning.
- Historical Git diff from original training revision must touch only audited non-training files (audit modules, scripts, tests, docs). If physics/training/reconstruction/asset changes are found, this readiness checker fails closed.
- The retrospective report is written as **a new file** under `D:\Projects\.archives\F1-RL\wp7-exploratory\audits`, outside Git and away from original run subfolders. The CLI never overwrites an existing report.

## Six hard blockers and how to resolve them

| Gate | Present result | Required explicit action |
|---|---|---|
| Physical body geometry | **Missing** in original `VehicleParameters` | Specify and version vehicle length, width, height, local chassis-centre offset, collision hull and tire footprint for the *intended* vehicle |
| Simulator collision validity | **Point-only** post-step obstacle occupancy | Either scope the scientific claim explicitly to a point-vehicle simulator **or** implement continuous swept collision in a new candidate, re-train and re-evaluate under prospective protocol |
| Right Turn/corner cutting | Original policy max lateral deviations **5.511796 m / 4.415475 m** | Declare acceptable route corridor/tolerances from track/safety intent, independent of observed policy outcome; rerun TRAIN acceptance under this pre-declared rule |
| Historical training environment provenance | `provenance_file = null` for both runs | Document this limitation, preserve hashes/Git revision and source inventory; require prospective dependency/host/version capture on any further training; cannot retroactively prove exact original host |
| Robustness | One deterministic episode/track/milestone | Register a repeated-episode and independent-seed protocol on TRAIN, with success/variance and CI thresholds before evaluating |
| Independent evidence backup | Models, manifests and audits share drive D | Verify an independent storage destination, checksum all copies and record backup manifest; do not report off-device safety until actually verified |

The `swept_3d_body_proxy` audit's large 5.0 × 2.2 × 1.4 m profile had calibration **−0.040417 m** conservative lower bound (inconclusive, *not* observed collision) and confirmation **+0.598965 m**. The proxy assumptions are not actual vehicle dimensions and therefore cannot independently resolve body geometry.

## Reproducible freeze gate

```powershell
cd D:\Projects\.worktrees\F1-RL\wp7-rebuild
.\.venv\Scripts\python.exe scripts\wp7_freeze_readiness.py --output "D:\Projects\.archives\F1-RL\wp7-exploratory\audits\wp7-freeze-readiness-next.json"
```

The runner verifies the original checkpoint/evaluation evidence, seeds, PPO/vector config, source fingerprints, original and derived hashes, no post-training simulator changes, and both geometry audit outcomes. It returns `scientific_freeze_permitted=false`, `heldout_evaluation_permitted=false`, and a typed list of blocking decisions.

**Approval boundary:** `review_required` is a positive evidence-integrity outcome with *outstanding acceptance gates*. No automatic promotion to `freeze_candidate` is legitimate. A future scientific freeze requires a separate explicit experiment definition, independent review, accepted risk scope, documented prospective evaluation, and authorization before opening HELD-OUT.

The static `pilot_preflight.py` warning text predates the two exploratory runs and should not be interpreted as a live scan of the artifact directories; the retrospective gate is the authoritative check for **archived runs**.

No HELD-OUT rollout, new PPO train/confirmation run, WP8, `main` merge or claim of scientifically verified unseen-track generalization has occurred in this stage.
