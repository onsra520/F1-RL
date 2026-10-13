"""Synthetic bytes only; no historical PPO archive or Google Drive upload."""
from copy import deepcopy
import json

import pytest

from f1rl.learning.evidence_backup import (
    build_evidence_manifest, verify_evidence_manifest, serialize_manifest,
)


def test_exact_manifest_and_restored_bytes(tmp_path):
    source=tmp_path/"source"
    destination=tmp_path/"restored"
    source.mkdir()
    (source/"models").mkdir()
    (source/"models"/"policy.zip").write_bytes(b"synthetic-checkpoint-bytes")
    (source/"train.json").write_text('{"tracks":"TRAIN"}',encoding="utf-8")
    report=build_evidence_manifest(source)
    assert report["algorithm"]=="sha256"
    assert sorted(x["path"] for x in report["entries"])==["models/policy.zip","train.json"]
    assert serialize_manifest(report)==serialize_manifest(report)
    destination.mkdir()
    (destination/"models").mkdir()
    (destination/"models"/"policy.zip").write_bytes(b"synthetic-checkpoint-bytes")
    (destination/"train.json").write_text('{"tracks":"TRAIN"}',encoding="utf-8")
    assert verify_evidence_manifest(destination,report)==2
    (destination/"models"/"policy.zip").write_bytes(b"different")
    with pytest.raises(ValueError,match="restore differs"):
        verify_evidence_manifest(destination,report)


def test_missing_and_extra_file_are_rejected(tmp_path):
    root=tmp_path/"evidence"
    root.mkdir()
    (root/"report.json").write_text("data")
    report=build_evidence_manifest(root)
    (root/"unregistered.txt").write_text("unregistered")
    with pytest.raises(ValueError,match="extra or missing"):
        verify_evidence_manifest(root,report)
    (root/"unregistered.txt").unlink()
    (root/"report.json").unlink()
    with pytest.raises(ValueError,match="restore differs"):
        verify_evidence_manifest(root,report)


@pytest.mark.parametrize("path",["../secret","/etc/passwd","C:\\secret","a/../b","a//b"])
def test_manifest_path_traversal_rejected(tmp_path,path):
    root=tmp_path/"restored"
    root.mkdir()
    report={"schema_version":1,"algorithm":"sha256",
            "entries":[{"path":path,"size_bytes":0,"sha256":"0"*64}]}
    with pytest.raises(ValueError,match="manifest|path|escapes"):
        verify_evidence_manifest(root,report)


def test_duplicate_manifest_records_rejected(tmp_path):
    root=tmp_path/"restored"
    root.mkdir()
    (root/"file.txt").write_bytes(b"test")
    report=build_evidence_manifest(root)
    duplicate=deepcopy(report)
    duplicate["entries"].append(deepcopy(report["entries"][0]))
    with pytest.raises(ValueError,match="duplicate"):
        verify_evidence_manifest(root,duplicate)


def test_empty_directory_does_not_count_as_backup(tmp_path):
    root=tmp_path/"source"
    root.mkdir()
    with pytest.raises(ValueError,match="no files"):
        build_evidence_manifest(root)


def test_manifest_does_not_accept_bools_as_sizes(tmp_path):
    root=tmp_path/"source"
    root.mkdir()
    (root/"file").write_bytes(b"binary")
    report=build_evidence_manifest(root)
    report["entries"][0]["size_bytes"]=True
    with pytest.raises(ValueError,match="size"):
        verify_evidence_manifest(root,report)


def test_corrupted_hash_does_not_verify(tmp_path):
    root=tmp_path/"source"
    root.mkdir()
    (root/"file").write_bytes(b"binary")
    report=build_evidence_manifest(root)
    report["entries"][0]["sha256"]="0"*64
    with pytest.raises(ValueError,match="restore differs"):
        verify_evidence_manifest(root,report)
