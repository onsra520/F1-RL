import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/restore_wp3_historical_evidence.py'


def test_restore_rejects_untrusted_archive_without_creating_destination(tmp_path):
    archive = tmp_path / 'bad.zip'
    archive.write_bytes(b'not the frozen archive')
    destination = tmp_path / 'restored'
    result = subprocess.run([sys.executable, str(SCRIPT), '--full', str(archive), '--continuation', str(archive), '--destination', str(destination)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'SHA-256' in result.stderr
    assert not destination.exists()


def test_restore_does_not_replace_existing_destination(tmp_path):
    destination = tmp_path / 'existing'
    destination.mkdir()
    marker = destination / 'keep'
    marker.write_text('unchanged')
    result = subprocess.run([sys.executable, str(SCRIPT), '--full', 'x', '--continuation', 'y', '--destination', str(destination)], capture_output=True, text=True)
    assert result.returncode != 0
    assert marker.read_text() == 'unchanged'
