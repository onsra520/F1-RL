# Script entrypoint migration map

All 48 command entrypoints now live in functional packages under `scripts/`. The former root-level names are listed as migration origins, not supported paths. Run grouped modules with `python -m`; direct file invocation is intentionally replaced because executables no longer modify `sys.path`.

The router preserves the existing aliases, forwarded arguments, exit codes, and each command’s own admission and evidence checks:

```powershell
python -m scripts.workflows list
python -m scripts.workflows evidence verify-frozen-assets
python -m scripts.workflows inference run-policy --help
```

| Stable command | Former root path | Current entrypoint | Scope |
| --- | --- | --- | --- |
| `diagnostics benchmark-environments` | `scripts/benchmark_vector_envs.py` | `scripts/diagnostics/benchmark_environments.py` | read-only |
| `inference package-handoff` | `scripts/build_demo_handoff.py` | `scripts/inference/package_handoff.py` | read-only |
| `evaluation compare-driving` | `scripts/compare_driving.py` | `scripts/evaluation/compare_driving.py` | development-execution |
| `diagnostics diagnose-driving` | `scripts/diagnose_driving.py` | `scripts/diagnostics/diagnose_driving.py` | development-execution |
| `evidence attest-source` | `scripts/engineering_attestation.py` | `scripts/evidence/engineering_attestation.py` | read-only |
| `training exploratory-policy` | `scripts/exploratory_ppo.py` | `scripts/training/exploratory_policy.py` | training-explicit |
| `training launch-exploration` | `scripts/launch_exploratory_demo.py` | `scripts/training/launch_demo.py` | training-explicit |
| `training local-simulation` | `scripts/local_demo.py` | `scripts/training/local_simulation.py` | training-explicit |
| `evidence restore-history` | `scripts/restore_wp3_historical_evidence.py` | `scripts/evidence/restore_historical_evidence.py` | read-only |
| `inference run-policy` | `scripts/run_inference.py` | `scripts/inference/run_policy.py` | development-execution |
| `inference smoke-frozen-policy` | `scripts/smoke_consolidated_wp3_model.py` | `scripts/inference/smoke_frozen_policy.py` | development-execution |
| `evidence verify-frozen-assets` | `scripts/verify_consolidated_wp3_assets.py` | `scripts/evidence/verify_frozen_assets.py` | read-only |
| `evidence verify-historical-records` | `scripts/verify_historical_source_records.py` | `scripts/evidence/verify_historical_records.py` | read-only |
| `diagnostics smoke-crash-gym` | `scripts/wp3_applied_crash_gym_smoke.py` | `scripts/diagnostics/crash_gym_smoke.py` | development-execution |
| `diagnostics smoke-crash-simulation` | `scripts/wp3_applied_crash_live_smoke.py` | `scripts/diagnostics/crash_simulation_smoke.py` | development-execution |
| `training campaign` | `scripts/wp3_campaign.py` | `scripts/training/campaign.py` | training-explicit |
| `evaluation current-admission-gate` | `scripts/wp3_current_eval_gate.py` | `scripts/evaluation/current_admission_gate.py` | read-only |
| `evaluation coordinate` | `scripts/wp3_eval_only_coordinator.py` | `scripts/evaluation/eval_only_coordinator.py` | restricted-execution |
| `evaluation preflight` | `scripts/wp3_eval_only_preflight.py` | `scripts/evaluation/eval_only_preflight.py` | read-only |
| `evaluation training-development` | `scripts/wp3_owner_train_dev_eval.py` | `scripts/evaluation/owner_train_dev_eval.py` | restricted-execution |
| `evaluation replay-preflight` | `scripts/wp3_physics_replay_preflight.py` | `scripts/evaluation/physics_replay_preflight.py` | read-only |
| `evidence register-crash-contract` | `scripts/wp3_register_crash_terminal_v1.py` | `scripts/evidence/register_crash_terminal_v1.py` | read-only |
| `evidence technical-acceptance` | `scripts/wp3_v4_technical_acceptance.py` | `scripts/evidence/technical_acceptance.py` | read-only |
| `evaluation replay-crash-development` | `scripts/wp3_v4_three_replay_dev_crash_contract.py` | `scripts/evaluation/three_replay_dev_crash_contract.py` | development-execution |
| `evaluation audit-replay-impulses` | `scripts/wp3_v4_three_replay_impulse_audit.py` | `scripts/evaluation/three_replay_impulse_audit.py` | development-execution |
| `evaluation replay-crash-contract` | `scripts/wp3_v4_three_replay_registered_crash_v1_corrected.py` | `scripts/evaluation/three_replay_registered_crash_v1.py` | development-execution |
| `evaluation replay-technical` | `scripts/wp3_v4_three_replay_technical_acceptance.py` | `scripts/evaluation/three_replay_technical_acceptance.py` | development-execution |
| `evidence verify-technical-acceptance` | `scripts/wp3_v4_verify_technical_acceptance.py` | `scripts/evidence/verify_technical_acceptance.py` | read-only |
| `evidence verify-crash-registration` | `scripts/wp3_verify_crash_terminal_v1.py` | `scripts/evidence/verify_crash_terminal_v1.py` | read-only |
| `evidence audit-swept-volume` | `scripts/wp7_audit_3d_swept_volume.py` | `scripts/evidence/audit_swept_volume.py` | read-only |
| `evidence audit-road-corridor` | `scripts/wp7_audit_right_turn_corridor.py` | `scripts/evidence/audit_road_corridor.py` | read-only |
| `evidence audit-feasibility` | `scripts/wp7_audit_train_feasibility.py` | `scripts/evidence/audit_train_feasibility.py` | read-only |
| `evidence audit-trajectories` | `scripts/wp7_audit_train_trajectories.py` | `scripts/evidence/audit_trajectories.py` | read-only |
| `diagnostics probe-gpu` | `scripts/wp7_capture_gpu.py` | `scripts/diagnostics/probe_gpu.py` | read-only |
| `diagnostics verify-planar-execution` | `scripts/wp7_certify_planar_execution.py` | `scripts/diagnostics/certify_planar_execution.py` | development-execution |
| `diagnostics core-readiness` | `scripts/wp7_core_readiness.py` | `scripts/diagnostics/core_readiness.py` | read-only |
| `evidence evidence-manifest` | `scripts/wp7_evidence_manifest.py` | `scripts/evidence/evidence_manifest.py` | read-only |
| `evaluation freeze-readiness` | `scripts/wp7_freeze_readiness.py` | `scripts/evaluation/freeze_readiness.py` | read-only |
| `evidence historical-source-audit` | `scripts/wp7_historical_source_audit.py` | `scripts/evidence/historical_source_audit.py` | read-only |
| `training pilot-preflight` | `scripts/wp7_pilot_preflight.py` | `scripts/training/pilot_preflight.py` | read-only |
| `internal ppo-fixture-worker` | `scripts/wp7_ppo_fixture_worker.py` | `scripts/internal/ppo_fixture_worker.py` | internal-only |
| `evaluation owner-review` | `scripts/wp7_review_owner_acceptance.py` | `scripts/evaluation/review_owner_acceptance.py` | read-only |
| `training exploratory-run` | `scripts/wp7_run_exploratory.py` | `scripts/training/run_exploratory.py` | training-explicit |
| `evaluation action-sensitivity` | `scripts/wp7_screen_train_action_sensitivity.py` | `scripts/evaluation/screen_train_action_sensitivity.py` | restricted-execution |
| `training train-acceptance` | `scripts/wp7_train_acceptance.py` | `scripts/training/train_acceptance.py` | read-only |
| `training training-admission` | `scripts/wp7_training_admission.py` | `scripts/training/training_admission.py` | read-only |
| `evidence verify-pilot` | `scripts/wp7_verify_pilot_evidence.py` | `scripts/evidence/verify_pilot_evidence.py` | read-only |
| `evidence verify-run` | `scripts/wp7_verify_run_evidence.py` | `scripts/evidence/verify_run.py` | read-only |

## Provenance handling

Former script names in archived research notes and immutable evidence records document the command as it existed when those records were created. They are historical text, not executable dependencies. No frozen model, evidence receipt, historical source record, or scientific admission record was rewritten. The new entrypoints retain protocol identifiers and scientific version labels only when they are part of the registered contract (for example, `crash_terminal_v1`).

Reusable implementation belongs in `src/`; package entrypoints call those modules and do not import another executable. Internal workers remain outside the public workflow dispatcher.

## Constrained broken reference

`.github/workflows/wp7-body-collision-regression.yml` still invokes the removed `scripts/wp7_review_owner_acceptance.py` path. This workflow was left untouched because the task explicitly excludes GitHub Actions. Update that command to `python -m scripts.evaluation.review_owner_acceptance` in a separately authorized change; no root compatibility file was retained.
