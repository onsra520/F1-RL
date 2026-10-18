"""Evidence-bound readiness report with an intentionally closed approval boundary.

No input boolean, matching checksum or green software receipt is an owner
signature. Review authentication is implemented separately; no trusted owner
bootstrap or independently verified scientific approval is present.
"""
from pathlib import Path
from hashlib import sha256
import os
import subprocess
import sys

from ..learning.owner_acceptance import review_preregistered_contract
from ..sandbox.tire_calibration_contract import REQUIRED_WP7_MEASUREMENTS
from .wp7_protocol import build_protocol, digest, _runtime, protocol_profile
from .wp7_track_feasibility import audit_train_tracks


class ScientificAdmissionError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _verify_software(root, protocol):
    paths = ["tests/integration/test_pv5_lifecycle.py", "tests/sandbox/test_pv5_coplanar_contact.py",
             "tests/sandbox/test_pv5_rotating_contact.py",
             "tests/research/test_pv5_track_feasibility.py", "tests/integration/test_pv5_outcomes.py",
             "tests/research/test_pv5_protocol.py", "tests/learning/training/test_pv5_backend_admission.py",
             "tests/integration/test_wp7_race_contract.py", "tests/integration/test_wp7_train_reset.py"]
    spatial = protocol_profile(protocol) in ('spatial_ground_v1','spatial_owner_v2','spatial_rotating_owner_v3')
    if spatial:
        paths.extend(['tests/research/test_wp7_spatial_profile.py','tests/research/test_wp7_spatial_profile_execution.py',
                      'tests/research/test_wp7_spatial_ramp_tracking.py'])
    if protocol_profile(protocol) in ('spatial_owner_v2','spatial_rotating_owner_v3'):
        paths.extend(['tests/research/test_wp7_owner_profile.py','tests/research/test_wp7_owner_profile_execution.py',
                      'tests/integration/test_hard_route_terminal_v2.py',
                      'tests/learning/test_hard_route_exit.py','tests/sandbox/test_spatial_compliant_lifecycle.py',
                      'tests/integration/test_spatial_compliant_gym.py'])
    if protocol_profile(protocol)=='spatial_rotating_owner_v3':
        paths.extend(['tests/research/test_wp7_rotating_owner_profile.py','tests/research/test_wp7_rotating_owner_execution.py',
                      'tests/sandbox/test_spatial_atlas_lifecycle.py','tests/sandbox/test_spatial_rotating_lifecycle.py'])
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", *paths, "-q"]
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("PYTEST_ADDOPTS", None)
    timeout_s=1200 if spatial else 120
    timed_out=False
    try:
        result=subprocess.run(command,cwd=root,env=environment,capture_output=True,text=True,timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        timed_out=True
        output=exc.stdout.decode(errors='replace') if isinstance(exc.stdout,bytes) else exc.stdout or ''
        error=exc.stderr.decode(errors='replace') if isinstance(exc.stderr,bytes) else exc.stderr or ''
        result=subprocess.CompletedProcess(command,124,output,error+f'\nsoftware verification exceeded {timeout_s}s')
    current = build_protocol(root,profile=protocol_profile(protocol))
    return {"command": command, "returncode": result.returncode, "timed_out":timed_out,"timeout_s":timeout_s,
            "source_unchanged": current["sha256"] == protocol["sha256"],
            "stdout_sha256": sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": sha256(result.stderr.encode()).hexdigest(),
            "output": result.stdout[-6000:], "stderr": result.stderr[-2000:],
            "scope": "focused_registered_software_checks_not_empirical_validation"}


def _verify_executed_train(root, protocol, output):
    from .wp7_train_acceptance import run_acceptance
    from .wp7_executed_feasibility import _execution_sources, _verify_module_origins
    from ..learning.pilot_preflight import TRAIN_IDS
    _verify_module_origins(root)
    summary = run_acceptance(root, output)
    expected = {(track, seed, 0, 0) for track in TRAIN_IDS for seed in range(3)}
    if (summary.get('sha256') != digest({k:v for k,v in summary.items() if k!='sha256'})
            or summary.get('schema_version') != 'wp7_train_execution_acceptance_v1'
            or summary.get('protocol_sha256') != protocol['sha256']
            or summary.get('source') != protocol['source'] or protocol['source']['dirty']
            or summary.get('source_files_sha256') != digest(_execution_sources(root))
            or summary.get('execution_profile') != 'spatial_rotating_owner_v3'
            or summary.get('source_unchanged') is not True
            or summary.get('complete') is not True
            or type(summary.get('all_pass')) is not bool
            or summary.get('training_authorized') is not False
            or summary.get('scientific_evidence') is not False
            or summary.get('scientific_acceptance') is not False
            or summary.get('permissions') != dict(training_permitted=False, heldout_permitted=False,
                 scientific_freeze_permitted=False, wp8_permitted=False)):
        raise ValueError('current-source non-authorizing execution evidence required')
    rows = summary.get('rows')
    if (not isinstance(rows, list) or len(rows) != 12
            or {(r['track_id'], r['paired_seed_index'], r['reset_rank'], r['reset_index'])
                for r in rows} != expected
            or summary['all_pass'] != all(r.get('status') == 'passed'
                and r.get('full_record_replay_equal') is True for r in rows)):
        raise ValueError('complete fixed TRAIN execution cohort required')
    if build_protocol(root, profile='spatial_rotating_owner_v3')['sha256'] != protocol['sha256']:
        raise ValueError('source/protocol changed during execution verification')
    _verify_module_origins(root)
    return dict(directory=str(Path(output).resolve()), summary=summary)


def evaluate_admission(repository_root, *, verify_software=False, capture_runtime=True,profile='pv5_v1',
                       run_manifest=None, review_envelope=None, verify_execution=False,
                       execution_output=None):
    if any(type(flag) is not bool for flag in (verify_software, capture_runtime, verify_execution)):
        raise ValueError("admission options must be bool")
    if verify_execution and execution_output is None:
        raise ValueError('new external execution output directory required')
    if verify_execution and profile != 'spatial_rotating_owner_v3':
        raise ValueError('execution verification requires registered rotating profile')
    if not verify_execution and execution_output is not None:
        raise ValueError('execution output requires verify_execution')
    root = Path(repository_root).resolve(strict=True)
    owner = review_preregistered_contract(root / "configs/recovery/wp7-owner-decisions-preregistered-v1.yml")
    protocol = build_protocol(root,profile=profile)
    execution_request = None
    review = None
    if review_envelope is not None and run_manifest is None:
        raise ValueError('run manifest required for authenticated review')
    if run_manifest is not None:
        from .wp7_run_manifest import verify_run_preparation
        execution_request = verify_run_preparation(run_manifest,
            expected_sha256=run_manifest.get('sha256'))
        if execution_request['protocol_sha256'] != protocol['sha256']:
            raise ValueError('run manifest protocol/source differs from admission')
        if review_envelope is not None:
            from .wp7_trusted_reviews import verify_repository_run_reviews
            review = verify_repository_run_reviews(root, review_envelope,
                protocol_sha256=protocol['sha256'], run_manifest_sha256=run_manifest['sha256'])
    track_audit = audit_train_tracks(root)
    runtime = _runtime() if capture_runtime else None
    software = _verify_software(root, protocol) if verify_software else None
    software_verified = software is not None and software["returncode"] == 0 and software["source_unchanged"]
    executed = _verify_executed_train(root, protocol, execution_output) if verify_execution else None
    execution_verified = executed is not None and executed['summary']['all_pass'] is True
    blockers = []
    def block(identity, layer, reason, evidence):
        blockers.append({"id": identity, "layer": layer, "reason": reason, "evidence": evidence})
    for reason in protocol["registration_blockers"]:
        block(reason, "registration", reason, "protocol.registration_blockers")
    rotating = profile == 'spatial_rotating_owner_v3'
    if rotating and not execution_verified:
        block('registered_execution_evidence_required', 'software',
              'verify current-source registered alternate trajectories, speed/stop and Gym execution; '
              'the historical authored sweep is a distinct finding', 'executed_train_evidence')
    elif not rotating:
        for row in track_audit["tracks"]:
            if row["geometry_status"] != "certified_authored_sweep":
                block(f"track_feasibility:{row['track_id']}", "track_feasibility",
                      "; ".join(row["blockers"]), f"track_audit.tracks:{row['track_id']}")
        block("speed_and_braking_protocol_missing", "track_feasibility",
              "geometric sweeps do not establish a dynamically feasible speed/braking schedule", "track_audit.tracks:braking")
    if protocol["source"]["dirty"]:
        block("source_tree_dirty", "provenance", "future training requires exact clean source", "protocol.source")
    if runtime is None:
        block("runtime_capture_required", "provenance", "no hardware probe requested for this denied entrypoint", "runtime")
    elif runtime["cpu"]["logical_processors"] is None or runtime["cpu"]["model"] == "unavailable":
        block("cpu_identity_incomplete", "provenance", "CPU capture incomplete", "runtime.cpu")
    if not software_verified:
        block("software_execution_evidence_required", "software",
              "source/test presence is not execution; use --verify-software and verify exact-HEAD CI separately", "software_verification")
    block("measured_calibration_missing", "physical_validation",
          "provisional engineering values lack measured references, uncertainty and independent validation",
          "required_measurements")
    if rotating:
        block('physical_domain_validation_scope_required', 'physical_validation',
              'bounded rotating/contact/recovery software exists; empirical validation scope, '
              'uncertainty and tolerances need independent agreement', 'protocol.registration.required_domains')
    else:
        block("general_contact_domain_incomplete", "physical_validation",
              "variable-omega TOI, mixed normals, tire patches and grounded obstacle recovery remain unsupported",
              "docs/research/wp7-pv5-transition-domains-v1.md")
    block("separate_approval_verifier_required", "authorization",
          "review signature verifier exists but trusted owner/reviewer identities and independent scientific admission are absent",
          "configs/research/wp7-trusted-reviewers-v1.json")
    block("owner_training_permission_false", "authorization",
          "preregistered owner reader does not grant training", "owner_permissions.training_permitted")
    document = {
        "schema_version": "wp7_admission_report_v1", "status": "physics_validation_blocked",
        "software_readiness": "software_verified" if software_verified else "execution_evidence_required",
        "software_verification": software,
        "executed_train_evidence": executed,
        "physical_validation_readiness": "blocked", "experiment_registration": "blocked",
        "authorization": {"contract": "separate_verified_approval_required_v1",
                          "verifier_implemented": True, "verified": False,
                          "run_review": review, "run_review_envelope": review_envelope},
        "execution_request": execution_request,
        "authored_track_audit_scope": 'historical_authored_sweep_not_alternate_execution',
        "training_authorized": False, "post_training_acceptance": "not_evaluated",
        "owner_permissions": {name: getattr(owner, name) for name in
            ("training_permitted", "heldout_permitted", "scientific_freeze_permitted", "wp8_permitted")},
        "evidence": {"protocol_sha256": protocol["sha256"], "source": protocol["source"],
                     "source_files": protocol["source_files"], "track_audit_sha256": digest(track_audit)},
        "required_measurements": list(REQUIRED_WP7_MEASUREMENTS),
        "runtime": runtime, "protocol": protocol, "track_audit": track_audit, "blockers": blockers,
    }
    return {**document, "sha256": digest(document)}


def reject_research_training(message, *, repository_root=None,profile='pv5_v1',run_manifest=None,
                             review_envelope=None):
    root = repository_root or Path(__file__).resolve().parents[2]
    try:
        report = evaluate_admission(root, capture_runtime=False,profile=profile,
                                    run_manifest=run_manifest, review_envelope=review_envelope)
    except Exception as exc:
        report = {"schema_version": "wp7_admission_report_v1", "status": "implementation_incomplete",
                  "training_authorized": False,
                  "blockers": [{"id": "admission_evidence_invalid", "reason": str(exc)}]}
    raise ScientificAdmissionError(message, report)
