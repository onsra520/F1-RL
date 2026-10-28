"""Saved untrained model inference in a spawned worker, never PPO learning."""
from hashlib import sha256
from pathlib import Path
import random
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

from src.learning.algorithms.ppo import PPOConfig, PPOModel
from src.research.wp7_cell_evaluation import execute_saved_ppo_fixture_cell
from src.research.wp7_protocol import build_protocol
from .test_evaluator import _factory

ROOT = Path(__file__).resolve().parents[3]


def saved_model(tmp_path):
    env = _factory(max_steps=2).create()
    try:
        model = PPOModel.create(env,PPOConfig(n_steps=2,batch_size=2,n_epochs=1,seed=7,device='cpu'))
        assert model.num_timesteps == 0
        path = model.save(tmp_path/'untrained.zip')
    finally:
        env.close()
    return path,sha256(path.read_bytes()).hexdigest()


def test_real_saved_ppo_stochastic_cell_replays_without_mutating_parent_rng(tmp_path):
    path,identity = saved_model(tmp_path)
    protocol = build_protocol(ROOT)
    args = dict(repository_root=ROOT,protocol=protocol,policy_id='multitrack-seed0',
        track_id='straight-sprint-v1',stratum='nominal',factory=_factory(max_steps=2),
        model_path=path,expected_model_sha256=identity,episodes=2)
    np.random.seed(123); random.seed(123); torch.manual_seed(123)
    before_np = np.random.get_state()
    before_random = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    first = execute_saved_ppo_fixture_cell(**args)
    second = execute_saved_ppo_fixture_cell(**args)
    assert first == second
    assert first['status'] == 'completed'
    assert first['worker']['isolation'] == 'spawn_process'
    assert first['scientific_evidence'] is False
    assert first['fixture']['configuration']['episode_config']['max_steps'] == 2
    assert first['fixture']['action_contract']['version'] == 'steering-throttle-brake0-v1'
    from src.track_assets.canonical import fingerprint_track_world
    assert first['fixture']['world_sha256'] == fingerprint_track_world(args['factory'].track)
    from src.research.wp7_protocol import digest
    fixture = first['fixture']
    assert fixture['sha256'] == digest({k:v for k,v in fixture.items() if k != 'sha256'})
    assert all(row['policy_action_count'] == 2 for row in first['episodes'])
    assert first['episodes'][0]['policy_rng_start_sha256'] != first['episodes'][1]['policy_rng_start_sha256']
    assert all(r['policy_rng_start_sha256'] != r['policy_rng_end_sha256'] for r in first['episodes'])
    assert np.array_equal(before_np[1],np.random.get_state()[1])
    assert before_np[2:] == np.random.get_state()[2:]
    assert random.getstate() == before_random
    assert torch.equal(torch.random.get_rng_state(),before_torch)


def test_wrong_model_identity_rejected_before_worker_start(tmp_path):
    path,identity = saved_model(tmp_path)
    with pytest.raises(ValueError,match='identity'):
        execute_saved_ppo_fixture_cell(repository_root=ROOT,protocol=build_protocol(ROOT),
            policy_id='multitrack-seed0',track_id='straight-sprint-v1',stratum='nominal',
            factory=_factory(),model_path=path,expected_model_sha256='0'*64)


def test_research_factory_rejected_before_model_access_even_if_subclass_lies(tmp_path):
    from src.learning.training.env_factory import F1RLEnvFactory
    from tests.integration.test_wp7_lifecycle_race import lifecycle_reset_factory
    class LyingFactory(F1RLEnvFactory):
        @property
        def requires_scientific_admission(self):
            return False
    candidate = lifecycle_reset_factory()
    bad = LyingFactory(**{f.name:getattr(candidate,f.name) for f in fields(candidate) if f.init})
    with pytest.raises(ValueError,match='research physics'):
        execute_saved_ppo_fixture_cell(repository_root=ROOT,protocol=build_protocol(ROOT),
            policy_id='multitrack-seed0',track_id='straight-sprint-v1',stratum='nominal',
            factory=bad,model_path=tmp_path/'never-read.zip',expected_model_sha256='0'*64)


def test_invalid_saved_model_aborts_once_and_preserves_evidence(tmp_path):
    path = tmp_path/'invalid.zip'
    path.write_bytes(b'not a model archive')
    record = execute_saved_ppo_fixture_cell(repository_root=ROOT,protocol=build_protocol(ROOT),
        policy_id='multitrack-seed0',track_id='straight-sprint-v1',stratum='nominal',
        factory=_factory(max_steps=2),model_path=path,
        expected_model_sha256=sha256(path.read_bytes()).hexdigest(),episodes=3)
    assert record['status'] == 'invalid_aborted'
    assert len(record['episodes']) == 1
    assert record['valid_episode_count'] == 0
    assert record['numerical_summary']['one_sided_cp95_lower'] is None
    assert record['numerical_summary']['meets_cell_numeric_criterion'] is False


def test_unavailable_explicit_cuda_never_silently_falls_back(tmp_path):
    if torch.cuda.is_available():
        pytest.skip('requires CPU-only host')
    with pytest.raises(RuntimeError,match='CUDA'):
        execute_saved_ppo_fixture_cell(repository_root=ROOT,protocol=build_protocol(ROOT),
            policy_id='multitrack-seed0',track_id='straight-sprint-v1',stratum='nominal',
            factory=_factory(),model_path=tmp_path/'never-read.zip',expected_model_sha256='0'*64,device='cuda')


def test_different_reward_implementations_with_same_fields_have_distinct_identity(tmp_path):
    from src.learning.rewards import RewardComposer, CollisionPenalty, OffTrackPenalty
    path,identity = saved_model(tmp_path)
    protocol = build_protocol(ROOT)
    records = []
    for term in (CollisionPenalty(1.),OffTrackPenalty(1.)):
        records.append(execute_saved_ppo_fixture_cell(repository_root=ROOT,protocol=protocol,
            policy_id='multitrack-seed0',track_id='straight-sprint-v1',stratum='nominal',
            factory=replace(_factory(max_steps=1),reward_composer=RewardComposer((term,))),
            model_path=path,expected_model_sha256=identity,episodes=1))
    assert records[0]['fixture']['sha256'] != records[1]['fixture']['sha256']
    assert records[0]['fixture']['reward_names'] == ['collision']
    assert records[1]['fixture']['reward_names'] == ['offtrack']
