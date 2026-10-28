import subprocess
import shutil
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


def test_restore_rejects_tampered_bundled_provenance(tmp_path):
    root = Path(__file__).resolve().parents[2]
    clone = tmp_path / 'checkout'
    (clone / 'scripts').mkdir(parents=True)
    shutil.copy2(SCRIPT, clone / 'scripts' / SCRIPT.name)
    docs = clone / 'docs/research'
    shutil.copytree(root / 'docs/research/evidence/wp3-historical-admission-inputs', docs / 'evidence/wp3-historical-admission-inputs')
    shutil.copy2(root / 'docs/research/FINAL-CONSOLIDATION-ASSETS.json', docs)
    path = docs / 'evidence/wp3-historical-admission-inputs/WP3-EVAL-SOURCE-PROVENANCE.json'
    path.write_bytes(path.read_bytes() + b' ')
    result = subprocess.run([sys.executable, str(clone / 'scripts' / SCRIPT.name), '--full', 'x', '--continuation', 'y', '--destination', str(tmp_path / 'result')], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'historical input SHA-256 mismatch' in result.stderr
    assert not (tmp_path / 'result').exists()
