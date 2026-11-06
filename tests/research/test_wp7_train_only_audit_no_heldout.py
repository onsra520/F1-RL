"""Regression: four-track admission audit must not open HELD-OUT geometry.

Previously audit_train_tracks called manifest loader without validate_asset_ids,
so thin handoff clones lacking evaluation geometry failed to produce a complete
DENIED scientific-admission report. TRAIN audit must authenticate TRAIN assets
and preserve all original HELD-OUT manifest metadata without opening it.
"""
from pathlib import Path
import pytest

from f1rl.learning.pilot_preflight import TRAIN_IDS
import f1rl.research.wp7_track_feasibility as feasibility

ROOT=Path(__file__).resolve().parents[2]


def test_owner_admission_train_audit_uses_exact_train4_validation_scope(monkeypatch):
    real=feasibility.load_track_manifest
    requested=[]
    def guarded_manifest(path,*,validate_asset_ids=None):
        requested.append(validate_asset_ids)
        assert validate_asset_ids==TRAIN_IDS
        return real(path,validate_asset_ids=validate_asset_ids)
    monkeypatch.setattr(feasibility,"load_track_manifest",guarded_manifest)
    result=feasibility.audit_train_tracks(ROOT)
    assert requested==[TRAIN_IDS]
    assert tuple(r["track_id"] for r in result["tracks"])==TRAIN_IDS
    assert result["training_authorized"] is False


def test_audit_never_reads_missing_heldout_geometry(monkeypatch):
    real=feasibility.load_track_asset
    opened=[]
    def guard(path):
        name=Path(path).as_posix()
        assert "/evaluation/" not in name,"HELD-OUT geometry must remain unread"
        opened.append(Path(path).name)
        return real(path)
    monkeypatch.setattr(feasibility,"load_track_asset",guard)
    result=feasibility.audit_train_tracks(ROOT)
    assert len(result["tracks"])==4
    assert opened and all("/evaluation/" not in x for x in opened)
