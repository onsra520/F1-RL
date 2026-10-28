import os
import subprocess
import sys
from pathlib import Path


def test_wp3_evidence_root_can_be_configured(tmp_path):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, F1RL_WP3_EVIDENCE=str(tmp_path))
    code = '''
import importlib.util, pathlib
from src.validation import wp3_current_eval_gate
assert wp3_current_eval_gate.OLD == pathlib.Path(%r)
for file in pathlib.Path("scripts").glob("wp3*.py"):
    text = file.read_text(encoding="utf8")
    if "F1RL_WP3_EVIDENCE" not in text:
        continue
    spec = importlib.util.spec_from_file_location("wp3_path_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert any(getattr(module, name, None) == pathlib.Path(%r) for name in ("HISTORY", "OLD", "E")), file
''' % (str(tmp_path), str(tmp_path))
    subprocess.run([sys.executable, '-c', code], cwd=root, env=env, check=True)
