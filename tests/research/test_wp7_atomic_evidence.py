import json

import pytest

from f1rl.research.wp7_cell_evaluation import save_cell_evidence
from f1rl.research.wp7_protocol import digest


def test_failed_publication_leaves_no_partial_file_or_temporary_file(tmp_path, monkeypatch):
    import f1rl.research.wp7_cell_evaluation as module
    record = dict(status='invalid_aborted', scientific_acceptance=False)
    record['sha256'] = digest(record)
    def fail(source, target):
        assert json.loads(source.read_text()) == record
        raise OSError('fixture publication failure')
    monkeypatch.setattr(module.os, 'link', fail)
    with pytest.raises(OSError):
        save_cell_evidence(record, tmp_path/'evidence.json')
    assert list(tmp_path.iterdir()) == []


def test_destination_created_during_publication_is_never_overwritten(tmp_path, monkeypatch):
    import f1rl.research.wp7_cell_evaluation as module
    real_link = module.os.link
    record = {'status':'completed'}
    record['sha256'] = digest(record)
    def race(source, target):
        target.write_bytes(b'previous evidence')
        real_link(source, target)
    monkeypatch.setattr(module.os, 'link', race)
    with pytest.raises(FileExistsError):
        save_cell_evidence(record, tmp_path/'evidence.json')
    assert (tmp_path/'evidence.json').read_bytes() == b'previous evidence'
    assert len(list(tmp_path.iterdir())) == 1
