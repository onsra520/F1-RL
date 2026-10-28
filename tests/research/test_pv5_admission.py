import json
from pathlib import Path

import pytest

from src.validation.wp7_admission import evaluate_admission, reject_research_training, ScientificAdmissionError

ROOT = Path(__file__).resolve().parents[2]


def test_report_uses_actual_track_and_source_evidence_and_separates_readiness():
    report = evaluate_admission(ROOT)
    assert report["training_authorized"] is False
    assert report["post_training_acceptance"] == "not_evaluated"
    assert report["owner_permissions"] == dict(training_permitted=False, heldout_permitted=False,
        scientific_freeze_permitted=False, wp8_permitted=False)
    assert "track_feasibility:right-turn-v1" in {b["id"] for b in report["blockers"]}
    assert len(report["evidence"]["protocol_sha256"]) == 64
    assert report["physical_validation_readiness"] == "blocked"
    assert report["authorization"]["verified"] is False
    assert json.loads(json.dumps(report)) == report


def test_absent_or_malformed_evidence_fails_closed_with_report(tmp_path):
    with pytest.raises(ScientificAdmissionError) as caught:
        reject_research_training("research physics training is blocked", repository_root=tmp_path)
    assert caught.value.report["training_authorized"] is False
    assert caught.value.report["blockers"][0]["id"] == "admission_evidence_invalid"


def test_hypothetical_green_software_cannot_override_owner_contract(monkeypatch):
    import src.validation.wp7_admission as admission
    monkeypatch.setattr(admission, "audit_train_tracks", lambda root: {"tracks": [], "manifest_sha256": "a"*64})
    report = admission.evaluate_admission(ROOT)
    assert report["authorization"]["verified"] is False
    assert report["training_authorized"] is False
    assert "separate_approval_verifier_required" in {b["id"] for b in report["blockers"]}


@pytest.mark.parametrize('partial',[b'partial test output','partial test output',None])
def test_software_timeout_returns_blocked_report_with_partial_diagnostics(monkeypatch,partial):
    import subprocess
    from types import SimpleNamespace
    import src.validation.wp7_admission as admission
    def timeout(command,**kwargs):
        budget=kwargs['timeout'];kwargs.clear()
        raise subprocess.TimeoutExpired(command,budget,output=partial,stderr=b'worker incomplete')
    monkeypatch.setattr(admission,'subprocess',SimpleNamespace(run=timeout,TimeoutExpired=subprocess.TimeoutExpired,
        CompletedProcess=subprocess.CompletedProcess))
    report=admission.evaluate_admission(ROOT,verify_software=True,capture_runtime=False,profile='spatial_rotating_owner_v3')
    evidence=report['software_verification']
    assert evidence['returncode']==124 and evidence['timed_out'] and evidence['timeout_s']==1200
    assert evidence['source_unchanged']
    assert 'worker incomplete' in evidence['stderr']
    assert ('partial test output' in evidence['output']) is (partial is not None)
    assert report['software_readiness']=='execution_evidence_required'
    assert not report['training_authorized'] and all(not value for value in report['owner_permissions'].values())
    assert json.loads(json.dumps(report))==report
