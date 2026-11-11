from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from f1rl.learning.algorithms import ppo
from f1rl.research import wp7_train5_v4_checkpoint as checkpoint
from scripts import wp7_penalty_trial_authorization as auth
from scripts import wp7_penalty_trial_runner as runner
from scripts import wp7_penalty_trial_evaluate as evaluation
from scripts.wp7_penalty_trial_protocol import build_proposal

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT.parent.parent / 'F1-RL-PHYSICS-V4-EVIDENCE-20261106'
pytestmark = pytest.mark.skipif(not EVIDENCE.is_dir() or not (ROOT.parent / 'slalom-fresh-audit.json').is_file() or not (ROOT.parent / 'TRAIN5-v4-internal-20261109' / 'evaluation-journal.jsonl').is_file(), reason='local trial witnesses or seed history are not available')


def test_fake_model_executes_paired_batch_without_signature(monkeypatch, tmp_path):
    proposal = deepcopy(build_proposal(ROOT))
    proposal['budget_per_run']['actual_final_steps'] = 16
    proposal['evaluation_seed_rows'] = [r for r in proposal['evaluation_seed_rows'] if r[2] == proposal['tracks'][0] and r[3] == 'nominal' and r[4] == 0]
    proposal['evaluation_episodes'] = 12
    monkeypatch.setattr(auth, 'validate_proposal', lambda root, value: None if value == proposal else (_ for _ in ()).throw(ValueError('proposal differs')))
    monkeypatch.setattr(runner, '_require_clean_source', lambda root: None)
    output = tmp_path / 'new-output'

    class FakeModel:
        def __init__(self, env): self.env, self.num_timesteps = env, 0
        def learn(self, total, progress_hook=None):
            self.env.reset()
            for _ in range(2):
                self.env.step(np.zeros((8, 3), dtype=np.float32))
                self.num_timesteps += 8
                if progress_hook is not None:
                    progress_hook(self.num_timesteps)
        def save(self, path):
            path.write_bytes(b'fake-model')
            return path

    def create_fake(env, config):
        ppo._validate_training_admission(env)
        return FakeModel(env)
    monkeypatch.setattr(ppo.PPOModel, 'create', create_fake)
    monkeypatch.setattr(checkpoint, '_prepare_v4_runner', lambda *a, **kw: SimpleNamespace(
        _config=SimpleNamespace(ppo=None, total_timesteps=16), _model_initializer=None,
        _final_model_observer=lambda path, steps: None))
    result = runner.run_trial(root=ROOT, proposal=proposal, output=output, evidence_root=EVIDENCE)
    assert len(result) == 12 and all(r['steps'] == 16 for r in result)
    assert len(list(output.glob('.claim-*'))) == 12 and (output / 'claim-batch').is_file()
    assert json.loads((output / 'training-result.json').read_text())['heldout_evaluated'] is False
    monkeypatch.setattr(evaluation.PPOModel, 'load', lambda *a, **kw: object())
    monkeypatch.setattr(evaluation, 'validate_loaded_policy', lambda *a: None)
    monkeypatch.setattr(evaluation, 'evaluate_one', lambda **kw: {'outcome': 'finished', 'valid_physics': True, 'episode_return': 1.0})
    monkeypatch.setattr(evaluation.os, 'fsync', lambda fd: None)
    report = evaluation.evaluate_trial(root=ROOT, output=output, proposal=proposal, evidence_root=EVIDENCE)
    assert report['episodes'] == 12 and report['heldout_evaluated'] is False
