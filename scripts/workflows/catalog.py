"""Stable workflow names for grouped executable modules.

The registry routes commands only; each module retains its own scientific and
training admission checks.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Workflow:
    group: str
    name: str
    script: str
    scope: str


_WORKFLOWS = (
    ('diagnostics', 'benchmark-environments', 'scripts/diagnostics/benchmark_environments.py', 'read-only'),
    ('evaluation', 'compare-driving', 'scripts/evaluation/compare_driving.py', 'development-execution'),
    ('diagnostics', 'diagnose-driving', 'scripts/diagnostics/diagnose_driving.py', 'development-execution'),
    ('evidence', 'attest-source', 'scripts/evidence/engineering_attestation.py', 'read-only'),
    ('training', 'exploratory-policy', 'scripts/training/exploratory_policy.py', 'training-explicit'),
    ('training', 'local-simulation', 'scripts/training/local_simulation.py', 'training-explicit'),
    ('evidence', 'restore-history', 'scripts/evidence/restore_historical_evidence.py', 'read-only'),
    ('inference', 'run-policy', 'scripts/inference/run_policy.py', 'development-execution'),
    ('inference', 'smoke-frozen-policy', 'scripts/inference/smoke_frozen_policy.py', 'development-execution'),
    ('evidence', 'verify-frozen-assets', 'scripts/evidence/verify_frozen_assets.py', 'read-only'),
    ('evidence', 'verify-historical-records', 'scripts/evidence/verify_historical_records.py', 'read-only'),
    ('diagnostics', 'smoke-crash-gym', 'scripts/diagnostics/crash_gym_smoke.py', 'development-execution'),
    ('diagnostics', 'smoke-crash-simulation', 'scripts/diagnostics/crash_simulation_smoke.py', 'development-execution'),
    ('training', 'campaign', 'scripts/training/campaign.py', 'training-explicit'),
    ('evaluation', 'current-admission-gate', 'scripts/evaluation/current_admission_gate.py', 'read-only'),
    ('evaluation', 'coordinate', 'scripts/evaluation/eval_only_coordinator.py', 'restricted-execution'),
    ('evaluation', 'preflight', 'scripts/evaluation/eval_only_preflight.py', 'read-only'),
    ('evaluation', 'training-development', 'scripts/evaluation/owner_train_dev_eval.py', 'restricted-execution'),
    ('evaluation', 'replay-preflight', 'scripts/evaluation/physics_replay_preflight.py', 'read-only'),
    ('evidence', 'register-crash-contract', 'scripts/evidence/register_crash_terminal_v1.py', 'read-only'),
    ('evidence', 'technical-acceptance', 'scripts/evidence/technical_acceptance.py', 'read-only'),
    ('evaluation', 'replay-crash-development', 'scripts/evaluation/three_replay_dev_crash_contract.py', 'development-execution'),
    ('evaluation', 'audit-replay-impulses', 'scripts/evaluation/three_replay_impulse_audit.py', 'development-execution'),
    ('evaluation', 'replay-crash-contract', 'scripts/evaluation/three_replay_registered_crash_v1.py', 'development-execution'),
    ('evaluation', 'replay-technical', 'scripts/evaluation/three_replay_technical_acceptance.py', 'development-execution'),
    ('evidence', 'verify-technical-acceptance', 'scripts/evidence/verify_technical_acceptance.py', 'read-only'),
    ('evidence', 'verify-crash-registration', 'scripts/evidence/verify_crash_terminal_v1.py', 'read-only'),
    ('evidence', 'audit-swept-volume', 'scripts/evidence/audit_swept_volume.py', 'read-only'),
    ('evidence', 'audit-road-corridor', 'scripts/evidence/audit_road_corridor.py', 'read-only'),
    ('evidence', 'audit-feasibility', 'scripts/evidence/audit_train_feasibility.py', 'read-only'),
    ('evidence', 'audit-trajectories', 'scripts/evidence/audit_trajectories.py', 'read-only'),
    ('diagnostics', 'probe-gpu', 'scripts/diagnostics/probe_gpu.py', 'read-only'),
    ('diagnostics', 'verify-planar-execution', 'scripts/diagnostics/certify_planar_execution.py', 'development-execution'),
    ('diagnostics', 'core-readiness', 'scripts/diagnostics/core_readiness.py', 'read-only'),
    ('evidence', 'evidence-manifest', 'scripts/evidence/evidence_manifest.py', 'read-only'),
    ('evaluation', 'freeze-readiness', 'scripts/evaluation/freeze_readiness.py', 'read-only'),
    ('evidence', 'historical-source-audit', 'scripts/evidence/historical_source_audit.py', 'read-only'),
    ('training', 'pilot-preflight', 'scripts/training/pilot_preflight.py', 'read-only'),
    ('internal', 'ppo-fixture-worker', 'scripts/internal/ppo_fixture_worker.py', 'internal-only'),
    ('evaluation', 'owner-review', 'scripts/evaluation/review_owner_acceptance.py', 'read-only'),
    ('training', 'exploratory-run', 'scripts/training/run_exploratory.py', 'training-explicit'),
    ('evaluation', 'action-sensitivity', 'scripts/evaluation/screen_train_action_sensitivity.py', 'restricted-execution'),
    ('training', 'train-acceptance', 'scripts/training/train_acceptance.py', 'read-only'),
    ('training', 'training-admission', 'scripts/training/training_admission.py', 'read-only'),
    ('evidence', 'verify-pilot', 'scripts/evidence/verify_pilot_evidence.py', 'read-only'),
    ('evidence', 'verify-run', 'scripts/evidence/verify_run.py', 'read-only'),
 )

WORKFLOWS = tuple(Workflow(*row) for row in _WORKFLOWS)
BY_COMMAND = {(item.group, item.name): item for item in WORKFLOWS}
if len(BY_COMMAND) != len(WORKFLOWS):
    raise ValueError("duplicate workflow command alias")
