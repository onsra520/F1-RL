# F1-RL WP1–WP7 remediation decision: 1B / 2B / 3B / 4A

**Decision owner:** project owner (user), 2026-10-13.
**Decision status:** adopted as TARGET REQUIREMENTS, **not** scientific freeze and not a declaration that any existing model is adequate.
**Historical evidence baseline:** `recovery/wp7-rebuild` @ `0cc4344` (untouched).
**Development chain:** `audit/h01-canonical-c2-contract` → `research/wp7-body-collision-v2` (prospective only).
**HELD-OUT:** SEALED. **WP8:** no execution.

## Confirmed scope

| Selection | Meaning | Binding acceptance condition |
|---|---|---|
| **1B** | Real rigid 3D body in simulator, not centre-only collision | Actual registered length/width/height/local centre; swept body vs 3D obstacles with translation and rotation; collision events, stopping, road contact and ramp behavior verified; no unreconciled model changes |
| **2B** | Finish + ordered checkpoints + legal route + corner-cutting control | Track corridor & road-union rules registered *before* training; validate **whole body footprint** and continuous interpolation; define airborne/ramp exception explicitly; no after-the-fact threshold tuned to prior PPO |
| **3B** | Measured robustness, not merely deterministic success | ≥3 independent PPO/sampling seed pairs, prospective randomized TRAIN-valid spawn/heading/obstacle scenarios, evaluation on fixed and varied TRAIN conditions, fixed statistical acceptance before inspecting results |
| **4A** | WP0 not formally defined | WP0 marked `not_applicable`. WP1 is earliest formal work package; do not manufacture a WP0 completion certificate |

## Registered DRAFT reliability gate

Stored in `configs/recovery/wp7-full-body-route-robustness-v1.yml`:

- Three independent PPO/sampling seed pairs minimum. Each is a genuine independent training initialization.
- Provisional evaluation budget: **60 stochastic-policy episodes per TRAIN track per policy**, plus predeclared deterministic episodes and randomized VALID TRAIN perturbations.
- **Exact one-sided Clopper–Pearson 95% lower success bound ≥0.90 on each TRAIN track for each independently trained policy** is proposed, but cannot be considered prospectively *final* until independent protocol review. 60 is an evaluation budget, not a guarantee the lower bound will pass.
- Environmental perturbations must be **feasibility-preserving** and sampled with predeclared ranges and seeds. `env.reset(seed)` currently does **not** vary environment geometry; genuine perturbation implementation and checks are required.
- Final training profile, evaluation splits, success definition, allowable body footprint excursion, route corridor and threshold ownership must all be frozen before final candidate training. **No HELD-OUT evaluations or model selection using HELD-OUT.**

Missing values intentionally `null`:
- Actual body `length_m`, `width_m`, `height_m`, `local_center_xyz_m`.
- Approved authored-route lateral deviation limit and exact airborne/ramp road-projection policy.
- Feasible TRAIN spawn displacement/heading perturbation ranges and validation procedure.

Values from retrospective `BodyProxy` sensitivity calculations must **not** silently become vehicle dimensions. Baseline's 3.0m wheelbase and 0.5m ride height do not specify body width/length/height.

## Stage 1 implemented on this review branch

- `f1rl/sandbox/body_collision.py`: prospective rigid-body `VehicleBodyGeometry`, complete 15-axis OBB SAT, shortest-arc quaternion interpolation, subdivision with conservative rigid-body displacement bound across timestep. Contact and resolution uncertainty both **block** the step.
- `step_world(..., body_geometry=...)` is opt-in; `Simulation`, `F1RLEnv` and `F1RLEnvFactory` expose geometry. No geometry supplied means **exact legacy engine**, not changed physics.
- `TrainingScenarioSet` requires equal body geometry across all participating training tracks; never mixes policies trained under different collision assumptions.
- `TrainingRunner` explicitly **refuses training when `body_geometry` is provided** until scientific candidate admission, geometry/corridor approval, provenance and independent seed protocol exist. Refusal occurs before new artifacts are created.
- Added regression tests for swept thin-obstacle tunneling, 3D rotation-only collision, rotated obstacles, local body centre, conservative unresolved intervals, legacy vs full-body behavior, and training admission failures.
- Cross-OS CI workflow runs dedicated physics tests and full regression. A passing test suite does **not** constitute scientific/physical validation.

## Stage 2 REQUIRED — WP1B/WP2/WP4/WP5 closure (NOT DONE)

1. Register actual physical body and tire footprint dimensions, reference pose and full-body collision identity in a distinct **v3 candidate protocol**. H-01's draft canonical C2 v2 is a mapping of the OLD C2 behavior and should never be relabeled as if it produced real body-aware PPO weights.
2. Implement correct whole-body track-surface union, ramp support/airborne semantics and continuous road-boundary checks. The new obstacle sweep alone is **not** sufficient for selected 2B.
3. Implement event-volume path entry for checkpoint/finish/boost/pickup where necessary, preserving ordered checkpoints; sampled-point event misses remain an unclosed issue.
4. Expose vehicle orientation and body pose in a versioned immutable scientific telemetry contract without modifying the 16D policy observation unless separately versioned.
5. Verify fixed-step determinism and body physics with property/metamorphic tests; define contact response, performance budget, and tunneling/false-positive rates under the registered geometry.

## Stage 3 REQUIRED — WP3/WP6/WP7 closure (NOT DONE)

1. Harden scientific training entrypoint with *immutable TRAIN allowlist*, world/content hashes, and zero access to HELD-OUT asset content before approval.
2. Require pre-run provenance (Git revision/clean state, training profile, package versions, OS, Python, track hashes, geometry and corpus identity, SEED pairs), then atomically write it before PPO instantiation. Ensure archival model/manifest validation and restoration from independent backup.
3. Implement genuine TRAIN environment variation, check spawn/heading/obstacle feasibility and preserve checkpoint ordering.
4. Train **new** independent models after final review. Old 200k models are legacy point-car baseline only.
5. Evaluate each predeclared policy with point-estimate and one-sided confidence bound; do not infer generalization from deterministic 4/4 or existing 128 action samples.
6. Re-audit WP1–WP7; close all high/medium findings with test evidence and reviewer decision; only then consider final scientific freeze. HELD-OUT is still sealed and WP8 has not begun.

## Constraints / evidence hygiene

Historical `wp7_freeze_readiness.py` rightly rejects changes to experimental source since the two 200k runs. **Do not whitelist** body-collision or new semantics changes as `AUDIT_ONLY_PATHS`; run archived evidence verifiers on the baseline code and validate new candidate only through a new separately versioned scientific gate.

The simulation checks continuous **assumed linear position and shortest-arc SLERP orientation** between discrete physics states. This is a conservative collision rule under that stated motion model, not full real-world mechanical dynamics. Conservatively unresolved intervals can create false-positive collision blocks; tune subdivision and performance by predeclared model requirements, **not** by retrofitting previously observed model success.

**Release verdict: BLOCKED.** No full-body PPO training, scientific freeze, HELD-OUT rollout, or WP8 initiation is approved.
