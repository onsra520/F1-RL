import hashlib

from src.interfaces import runner


def test_campaign_runner_uses_shared_artifact_hash_without_changing_sha256(tmp_path):
    artifact = tmp_path / "checkpoint.zip"
    artifact.write_bytes(b"campaign checkpoint fixture")

    assert runner._hash_file(artifact) == hashlib.sha256(artifact.read_bytes()).hexdigest()
