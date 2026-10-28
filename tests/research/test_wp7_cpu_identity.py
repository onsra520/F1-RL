"""CPU identity must use actual OS evidence, never an architecture placeholder."""
from pathlib import Path
import pytest
import src.research.wp7_protocol as protocol


def test_linux_cpu_model_uses_proc_when_platform_probe_is_empty(monkeypatch):
    monkeypatch.setattr(protocol.platform, 'processor', lambda: '')
    monkeypatch.setattr(protocol.platform, 'system', lambda: 'Linux')
    monkeypatch.delenv('PROCESSOR_IDENTIFIER', raising=False)
    monkeypatch.setattr(Path, 'read_text', lambda self, **kw: 'processor : 0\nmodel name : Measured CPU model\n')
    assert protocol._cpu_model() == 'Measured CPU model'


def test_cpu_model_preserves_native_probe_and_windows_environment(monkeypatch):
    monkeypatch.setattr(protocol.platform, 'processor', lambda: 'Native CPU')
    assert protocol._cpu_model() == 'Native CPU'
    monkeypatch.setattr(protocol.platform, 'processor', lambda: '')
    monkeypatch.setattr(protocol.platform, 'system', lambda: 'Windows')
    monkeypatch.setenv('PROCESSOR_IDENTIFIER', 'Windows CPU')
    assert protocol._cpu_model() == 'Windows CPU'


@pytest.mark.parametrize('text', ['', 'processor : 0\nmodel : 183\n', 'model name :   \n'])
def test_missing_cpu_model_remains_explicitly_unavailable(monkeypatch, text):
    monkeypatch.setattr(protocol.platform, 'processor', lambda: '')
    monkeypatch.setattr(protocol.platform, 'system', lambda: 'Linux')
    monkeypatch.delenv('PROCESSOR_IDENTIFIER', raising=False)
    monkeypatch.setattr(Path, 'read_text', lambda self, **kw: text)
    assert protocol._cpu_model() == 'unavailable'


def test_unreadable_proc_does_not_fabricate_cpu_identity(monkeypatch):
    monkeypatch.setattr(protocol.platform, 'processor', lambda: '')
    monkeypatch.setattr(protocol.platform, 'system', lambda: 'Linux')
    monkeypatch.delenv('PROCESSOR_IDENTIFIER', raising=False)
    def denied(self, **kwargs):
        raise OSError('unavailable OS source')
    monkeypatch.setattr(Path, 'read_text', denied)
    assert protocol._cpu_model() == 'unavailable'
