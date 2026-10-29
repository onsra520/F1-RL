import subprocess
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = 'scripts.evidence.restore_historical_evidence'


def test_restore_rejects_untrusted_archive_without_creating_destination(tmp_path):
    archive = tmp_path / 'bad.zip'
    archive.write_bytes(b'not the frozen archive')
    destination = tmp_path / 'restored'
    result = subprocess.run([sys.executable, '-m', MODULE, '--full', str(archive), '--continuation', str(archive), '--destination', str(destination)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'SHA-256' in result.stderr
    assert not destination.exists()


def test_restore_does_not_replace_existing_destination(tmp_path):
    destination = tmp_path / 'existing'
    destination.mkdir()
    marker = destination / 'keep'
    marker.write_text('unchanged')
    result = subprocess.run([sys.executable, '-m', MODULE, '--full', 'x', '--continuation', 'y', '--destination', str(destination)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert marker.read_text() == 'unchanged'


def test_restore_rejects_tampered_bundled_provenance(tmp_path):
    root = Path(__file__).resolve().parents[2]
    clone = tmp_path / 'checkout'
    (clone / 'scripts' / 'evidence').mkdir(parents=True)
    shutil.copy2(ROOT / 'scripts/__init__.py', clone / 'scripts/__init__.py')
    shutil.copy2(ROOT / 'scripts/evidence/__init__.py', clone / 'scripts/evidence/__init__.py')
    shutil.copy2(ROOT / 'scripts/evidence/restore_historical_evidence.py', clone / 'scripts/evidence/restore_historical_evidence.py')
    shutil.copytree(root / 'src', clone / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    docs = clone / 'docs/research'
    shutil.copytree(root / 'docs/research/evidence/wp3-historical-admission-inputs', docs / 'evidence/wp3-historical-admission-inputs')
    shutil.copy2(root / 'docs/research/FINAL-CONSOLIDATION-ASSETS.json', docs)
    path = docs / 'evidence/wp3-historical-admission-inputs/WP3-EVAL-SOURCE-PROVENANCE.json'
    path.write_bytes(path.read_bytes() + b' ')
    result = subprocess.run([sys.executable, '-m', MODULE, '--full', 'x', '--continuation', 'y', '--destination', str(tmp_path / 'result')], cwd=clone, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'historical input SHA-256 mismatch' in result.stderr
    assert not (tmp_path / 'result').exists()
