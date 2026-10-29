# Script compatibility and naming map

This migration map is based on the 48 Python entrypoints present at source baseline `b11a501`. **All old paths are retained unchanged** because external handoffs, subprocesses, audit allow-lists and historic provenance may depend on their spelling. The new workflow interface supplies stable **versionless command aliases**. It does not bypass the existing entrypoint's admission controls or turn historical execution into fresh scientific evidence.

Usage: `python -m scripts.workflows <group> <action> [original script arguments...]`

Each table row maps one alias to one executable. `internal-only` is intentionally not invokable through this public dispatcher.

## diagnostics

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `diagnostics benchmark-environments` | `scripts/benchmark_vector_envs.py` | read-only |
| `diagnostics diagnose-driving` | `scripts/diagnose_driving.py` | development-execution |
| `diagnostics smoke-crash-gym` | `scripts/wp3_applied_crash_gym_smoke.py` | development-execution |
| `diagnostics smoke-crash-simulation` | `scripts/wp3_applied_crash_live_smoke.py` | development-execution |
| `diagnostics probe-gpu` | `scripts/wp7_capture_gpu.py` | read-only |
| `diagnostics verify-planar-execution` | `scripts/wp7_certify_planar_execution.py` | development-execution |
| `diagnostics core-readiness` | `scripts/wp7_core_readiness.py` | read-only |

## evaluation

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `evaluation compare-driving` | `scripts/compare_driving.py` | development-execution |
| `evaluation current-admission-gate` | `scripts/wp3_current_eval_gate.py` | read-only |
| `evaluation coordinate` | `scripts/wp3_eval_only_coordinator.py` | restricted-execution |
| `evaluation preflight` | `scripts/wp3_eval_only_preflight.py` | read-only |
| `evaluation training-development` | `scripts/wp3_owner_train_dev_eval.py` | restricted-execution |
| `evaluation replay-preflight` | `scripts/wp3_physics_replay_preflight.py` | read-only |
| `evaluation replay-crash-development` | `scripts/wp3_v4_three_replay_dev_crash_contract.py` | development-execution |
| `evaluation audit-replay-impulses` | `scripts/wp3_v4_three_replay_impulse_audit.py` | development-execution |
| `evaluation replay-crash-contract` | `scripts/wp3_v4_three_replay_registered_crash_v1_corrected.py` | development-execution |
| `evaluation replay-technical` | `scripts/wp3_v4_three_replay_technical_acceptance.py` | development-execution |
| `evaluation freeze-readiness` | `scripts/wp7_freeze_readiness.py` | read-only |
| `evaluation owner-review` | `scripts/wp7_review_owner_acceptance.py` | read-only |
| `evaluation action-sensitivity` | `scripts/wp7_screen_train_action_sensitivity.py` | restricted-execution |

## evidence

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `evidence attest-source` | `scripts/engineering_attestation.py` | read-only |
| `evidence restore-history` | `scripts/restore_wp3_historical_evidence.py` | read-only |
| `evidence verify-frozen-assets` | `scripts/verify_consolidated_wp3_assets.py` | read-only |
| `evidence verify-historical-records` | `scripts/verify_historical_source_records.py` | read-only |
| `evidence register-crash-contract` | `scripts/wp3_register_crash_terminal_v1.py` | read-only |
| `evidence technical-acceptance` | `scripts/wp3_v4_technical_acceptance.py` | read-only |
| `evidence verify-technical-acceptance` | `scripts/wp3_v4_verify_technical_acceptance.py` | read-only |
| `evidence verify-crash-registration` | `scripts/wp3_verify_crash_terminal_v1.py` | read-only |
| `evidence audit-swept-volume` | `scripts/wp7_audit_3d_swept_volume.py` | read-only |
| `evidence audit-road-corridor` | `scripts/wp7_audit_right_turn_corridor.py` | read-only |
| `evidence audit-feasibility` | `scripts/wp7_audit_train_feasibility.py` | read-only |
| `evidence audit-trajectories` | `scripts/wp7_audit_train_trajectories.py` | read-only |
| `evidence evidence-manifest` | `scripts/wp7_evidence_manifest.py` | read-only |
| `evidence historical-source-audit` | `scripts/wp7_historical_source_audit.py` | read-only |
| `evidence verify-pilot` | `scripts/wp7_verify_pilot_evidence.py` | read-only |
| `evidence verify-run` | `scripts/wp7_verify_run_evidence.py` | read-only |

## inference

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `inference package-handoff` | `scripts/build_demo_handoff.py` | read-only |
| `inference run-policy` | `scripts/run_inference.py` | development-execution |
| `inference smoke-frozen-policy` | `scripts/smoke_consolidated_wp3_model.py` | development-execution |

## internal

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `internal ppo-fixture-worker` | `scripts/wp7_ppo_fixture_worker.py` | internal-only |

## training

| Stable command | Existing compatibility path | Execution scope |
| --- | --- | --- |
| `training exploratory-policy` | `scripts/exploratory_ppo.py` | training-explicit |
| `training launch-exploration` | `scripts/launch_exploratory_demo.py` | training-explicit |
| `training local-simulation` | `scripts/local_demo.py` | training-explicit |
| `training campaign` | `scripts/wp3_campaign.py` | training-explicit |
| `training pilot-preflight` | `scripts/wp7_pilot_preflight.py` | read-only |
| `training exploratory-run` | `scripts/wp7_run_exploratory.py` | training-explicit |
| `training train-acceptance` | `scripts/wp7_train_acceptance.py` | read-only |
| `training training-admission` | `scripts/wp7_training_admission.py` | read-only |

## Rules for subsequent physical moves

1. Search all scripts, tests, configs, docs, subprocess calls, worker pools and immutable source-binding manifests before moving a compatibility path.
2. Historical IDs and immutable evidence paths retain their recorded bytes and spellings. A current engineering attestation is **not** a replacement for historical source identity.
3. Move reusable implementation into `src/` before deprecating old wrappers; preserve script-specific arguments, stdout, exit codes and scientific fail-closed checks.
4. Record any removed path with consumers, compatibility window, tests and a deliberate migration decision. No path in this baseline map is deleted by this change.
