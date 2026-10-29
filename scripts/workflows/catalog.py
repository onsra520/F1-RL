"""Stable workflow names for existing standalone entrypoints.

Legacy script paths remain unchanged while historical provenance and callers are audited.
The registry is a routing table, not a scientific or training authorization layer.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Workflow:
    group: str
    name: str
    script: str
    scope: str


_WORKFLOWS = (
    ("diagnostics", "benchmark-environments", "benchmark_vector_envs.py", "read-only"),
    ("inference", "package-handoff", "build_demo_handoff.py", "read-only"),
    ("evaluation", "compare-driving", "compare_driving.py", "development-execution"),
    ("diagnostics", "diagnose-driving", "diagnose_driving.py", "development-execution"),
    ("evidence", "attest-source", "engineering_attestation.py", "read-only"),
    ("training", "exploratory-policy", "exploratory_ppo.py", "training-explicit"),
    ("training", "launch-exploration", "launch_exploratory_demo.py", "training-explicit"),
    ("training", "local-simulation", "local_demo.py", "training-explicit"),
    ("evidence", "restore-history", "restore_wp3_historical_evidence.py", "read-only"),
    ("inference", "run-policy", "run_inference.py", "development-execution"),
    ("inference", "smoke-frozen-policy", "smoke_consolidated_wp3_model.py", "development-execution"),
    ("evidence", "verify-frozen-assets", "verify_consolidated_wp3_assets.py", "read-only"),
    ("evidence", "verify-historical-records", "verify_historical_source_records.py", "read-only"),
    ("diagnostics", "smoke-crash-gym", "wp3_applied_crash_gym_smoke.py", "development-execution"),
    ("diagnostics", "smoke-crash-simulation", "wp3_applied_crash_live_smoke.py", "development-execution"),
    ("training", "campaign", "wp3_campaign.py", "training-explicit"),
    ("evaluation", "current-admission-gate", "wp3_current_eval_gate.py", "read-only"),
    ("evaluation", "coordinate", "wp3_eval_only_coordinator.py", "restricted-execution"),
    ("evaluation", "preflight", "wp3_eval_only_preflight.py", "read-only"),
    ("evaluation", "training-development", "wp3_owner_train_dev_eval.py", "restricted-execution"),
    ("evaluation", "replay-preflight", "wp3_physics_replay_preflight.py", "read-only"),
    ("evidence", "register-crash-contract", "wp3_register_crash_terminal_v1.py", "read-only"),
    ("evidence", "technical-acceptance", "wp3_v4_technical_acceptance.py", "read-only"),
    ("evaluation", "replay-crash-development", "wp3_v4_three_replay_dev_crash_contract.py", "development-execution"),
    ("evaluation", "audit-replay-impulses", "wp3_v4_three_replay_impulse_audit.py", "development-execution"),
    ("evaluation", "replay-crash-contract", "wp3_v4_three_replay_registered_crash_v1_corrected.py", "development-execution"),
    ("evaluation", "replay-technical", "wp3_v4_three_replay_technical_acceptance.py", "development-execution"),
    ("evidence", "verify-technical-acceptance", "wp3_v4_verify_technical_acceptance.py", "read-only"),
    ("evidence", "verify-crash-registration", "wp3_verify_crash_terminal_v1.py", "read-only"),
    ("evidence", "audit-swept-volume", "wp7_audit_3d_swept_volume.py", "read-only"),
    ("evidence", "audit-road-corridor", "wp7_audit_right_turn_corridor.py", "read-only"),
    ("evidence", "audit-feasibility", "wp7_audit_train_feasibility.py", "read-only"),
    ("evidence", "audit-trajectories", "wp7_audit_train_trajectories.py", "read-only"),
    ("diagnostics", "probe-gpu", "wp7_capture_gpu.py", "read-only"),
    ("diagnostics", "verify-planar-execution", "wp7_certify_planar_execution.py", "development-execution"),
    ("diagnostics", "core-readiness", "wp7_core_readiness.py", "read-only"),
    ("evidence", "evidence-manifest", "wp7_evidence_manifest.py", "read-only"),
    ("evaluation", "freeze-readiness", "wp7_freeze_readiness.py", "read-only"),
    ("evidence", "historical-source-audit", "wp7_historical_source_audit.py", "read-only"),
    ("training", "pilot-preflight", "wp7_pilot_preflight.py", "read-only"),
    ("internal", "ppo-fixture-worker", "wp7_ppo_fixture_worker.py", "internal-only"),
    ("evaluation", "owner-review", "wp7_review_owner_acceptance.py", "read-only"),
    ("training", "exploratory-run", "wp7_run_exploratory.py", "training-explicit"),
    ("evaluation", "action-sensitivity", "wp7_screen_train_action_sensitivity.py", "restricted-execution"),
    ("training", "train-acceptance", "wp7_train_acceptance.py", "read-only"),
    ("training", "training-admission", "wp7_training_admission.py", "read-only"),
    ("evidence", "verify-pilot", "wp7_verify_pilot_evidence.py", "read-only"),
    ("evidence", "verify-run", "wp7_verify_run_evidence.py", "read-only"),
)

WORKFLOWS = tuple(Workflow(*row) for row in _WORKFLOWS)
BY_COMMAND = {(item.group, item.name): item for item in WORKFLOWS}
if len(BY_COMMAND) != len(WORKFLOWS):
    raise ValueError("duplicate workflow command alias")


# Primary module entrypoints for commands whose implementation has been extracted.
# Historical, source-bound commands are deliberately not rewritten or relocated.
PRIMARY_PATHS = {
    ("diagnostics", "benchmark-environments"): "scripts/diagnostics/benchmark_environments.py",
    ("inference", "package-handoff"): "scripts/inference/package_handoff.py",
    ("evaluation", "compare-driving"): "scripts/evaluation/compare_driving.py",
    ("diagnostics", "diagnose-driving"): "scripts/diagnostics/diagnose_driving.py",
    ("training", "local-simulation"): "scripts/training/local_simulation.py",
    ("inference", "run-policy"): "scripts/inference/run_policy.py",
    ("evidence", "verify-frozen-assets"): "scripts/evidence/verify_frozen_assets.py",
    ("evidence", "audit-swept-volume"): "scripts/evidence/audit_swept_volume.py",
    ("evidence", "audit-road-corridor"): "scripts/evidence/audit_road_corridor.py",
    ("evidence", "audit-trajectories"): "scripts/evidence/audit_trajectories.py",
    ("diagnostics", "probe-gpu"): "scripts/diagnostics/probe_gpu.py",
    ("evidence", "evidence-manifest"): "scripts/evidence/evidence_manifest.py",
    ("evidence", "historical-source-audit"): "scripts/evidence/historical_source_audit.py",
    ("evidence", "verify-run"): "scripts/evidence/verify_run.py",
}
