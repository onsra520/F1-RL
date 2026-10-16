# F1-RL core-first readiness: 90% functional code + compatibility closure

Scope: Python simulation, physics, perception, Gym, training, evaluation, track/race rules, provenance, and scientific admission. **FastAPI, WebSocket, React and Three.js are excluded.**

This is an engineering delivery target, NOT a verified completion percentage, a scientific sign-off, or permission for PPO training. Avoid treating test counts as percentage complete.

## Definition of 90% for EACH subsystem

Before assigning any completion percentage, enumerate its registered, user-visible capabilities and risk-weighted acceptance cases in a traceability matrix. The **core functionality target** is >=90% accepted, with **100% of P0/safety-critical contracts accepted**; a missing core pathway disqualifies the subsystem regardless of a weighted score. Every accepted case requires executable code reached through the actual runtime entrypoint, a deterministic reproducible test, and relevant negative/fail-closed coverage.

The remaining effort is a dedicated **compatibility and integration pass**, not license to leave unsafe behavior or missing physical prerequisites. It includes legacy C2 reproducibility, Windows/Linux and CPU/CUDA support where available, deterministic seed replay, cross-module contracts, and mixed supported/unsupported scenario handling.

**Scientific empirical validation remains a separate binary gate**: no percentage or number of synthetic tests can substitute for physical measurements, independent replication, private Drive restore, owner admission, or HELD-OUT controls.

## Subsystem acceptance inventory

| Subsystem | Implementation baseline | P0 functional exit requirement (90% candidate) | Compatibility exit requirement |
|---|---|---|---|
| Simulator state/runtime | Fixed-step Simulation, snapshots, versioned configs | Ground/airborne transitions share correct state and event ordering; no unregistered transition or state loss | Historical state/step behavior unchanged; Windows/Linux deterministic evidence |
| Vehicle dynamics/contact | Suspension, bounded flat coplanar lifecycle, contact impulse audits | Required TRAIN track domains have correctly bracketed impact/recovery or explicit registered exclusions; no ambiguous physics counted valid | No silent fallback; known mixed normals/variable omega rejected; numerical portability |
| Road/track feasibility | Versioned TRAIN assets + body-aware authored-path audit | All required TRAIN courses have a certified feasible swept-body trajectory and gate passage; speed/braking assumptions registered | Source/world hashes, geometry tolerance, old tracks unchanged |
| Race/episode semantics | Grounded directed gates, route 2/4m, research outcome audit | Supported lifecycle crossings, invalid vs legitimate terminal failure, finish/timeout precedence have approved versioned contract and tests | Baseline C2 6m/10 steps and reward semantics reproduced |
| Perception/control | Rangefinders, route observation, steering/throttle | Observations/actions valid in every supported physical phase; action space limitations (brake=0) explicit and evaluated | Observation shape and older models remain compatible where specified |
| TRAIN factory/vector PPO | Gym/SB3 runner, scenarios, per-reset perturbation | Per-replicate PPO/sampling/perturbation seeds bind consistently across factory, vector ranks and training/evaluation reconstruction; no unauthorized learn | DummyVecEnv/SubprocVecEnv, Win spawn, save/load/reattach guards |
| Evaluation/statistics | Episode metrics, paired-seed protocol draft, CP checker | True per-policy/track/stratum evaluation path, genuine >=60 trials, CP95 >=0.90 check, invalid samples not silently pooled; **execution stays blocked** until approved | Replays and serialized metrics schema explicit; held-out sealed |
| Provenance & admission | Git/content hashes, runtime manifests, fail-closed report | Verifiable end-to-end evidence manifest, scoped approvals and independent auditor; green tests cannot self-authorize | Clean checkout/restoration, cross-host validation, artifact hash verification |

## Work sequence

1. **Fix verified code gaps first**: policy-specific seed reconstruction, grounded/lifecycle race-reset handoff, event ordering, per-cell evaluator wiring.
2. **Resolve physical/track-domain mismatches**: slalom bounded-turn candidates; right-turn road/gate geometry; ramp slope continuity. Audit alternatives without rewriting canonical assets.
3. **Close the approved episode contracts**: route hard-violation failure is a distinct owner decision, not automatically a physics error or reward penalty.
4. **Audit full research execution**: exact registration, provenance, genuine evaluation and independent verifiers; no unauthorized training.
5. **Compatibility sweep**: C2 untouched, both OS CI, release/packaging, CPU/CUDA where possible, subprocess spawn, serial formats and cross-version validation.

## Initial verified gap-closure increment

PR #20 reconstructs every TRAIN scenario with perturbation seed 1000 independent of which paired replicate is registered. The new policy-aware preparation API requires an explicit seed index and creates a policy-specific scenario set, with a machine-readable identity receipt. It does not call PPO or alter the admission decision. Its tests must verify three seed replicates and prevent implicit seed-zero fallback.

## Current restrictions

Do not access HELD-OUT, run new research PPO while admission is false, alter owner choices, claim a general 6-DOF solver from bounded fixtures, fabricate empirical calibration, modify historical checkpoints, or merge stacked research PRs without review.

**Exit decision**: Until all applicable P0 cases and required independent evidence are satisfied, subsystem readiness is **in progress**, not >=90% complete.
