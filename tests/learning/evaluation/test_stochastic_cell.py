"""Stochastic evaluator plumbing is tested with software-only legacy fixtures."""
from pathlib import Path

import numpy as np
import pytest

from f1rl.research.wp7_admission import ScientificAdmissionError
from f1rl.research.wp7_cell_evaluation import (
    evaluate_registered_cell, execute_software_fixture_cell, policy_stochastic_seed,
)
from f1rl.research.wp7_protocol import build_protocol, evaluation_seed
from .mocks.policies import ConstantPolicy, FailingPolicy
from .test_evaluator import _factory

ROOT = Path(__file__).resolve().parents[3]


def _protocol():
    return build_protocol(ROOT)


def test_stochastic_fixture_cell_separates_policy_and_environment_rng(tmp_path):
    protocol = _protocol()
    path = tmp_path / "fixture.zip"
    path.write_bytes(b"model identity fixture; not real trained PPO")
    policies = []
    def factory(seed):
        policy = ConstantPolicy(np.array([0.0, float(np.random.default_rng(seed).random())], dtype=np.float32))
        policies.append(policy)
        return policy
    args = dict(
        repository_root=ROOT, protocol=protocol, policy_id="multitrack-seed0",
        track_id="straight-sprint-v1", stratum="nominal", factory=_factory(max_steps=2),
        model_path=path, policy_for_seed=factory,
        episodes=2,
    )
    first = execute_software_fixture_cell(**args)
    second = execute_software_fixture_cell(**args)
    assert first == second
    assert first["sha256"] == second["sha256"]
    assert first["status"] == "completed"
    assert first["valid_episode_count"] == 2
    assert first["scientific_evidence"] is False
    assert all(p.deterministic_calls == [False, False] for p in policies)
    for index, row in enumerate(first["episodes"]):
        assert row["environment_seed"] == evaluation_seed(protocol, "multitrack-seed0", "straight-sprint-v1", "nominal", index)
        assert row["policy_rng_seed"] == policy_stochastic_seed(protocol, "multitrack-seed0", "straight-sprint-v1", "nominal", index)
        assert row["policy_rng_seed"] != row["environment_seed"]
        assert row["valid"] is True


def test_invalid_episode_aborts_cell_and_preserves_diagnostic(tmp_path):
    from f1rl.learning.evaluation.evaluator import EvaluationPolicy
    protocol = _protocol()
    path = tmp_path / "fixture.zip"
    path.write_bytes(b"fixture")
    cell = execute_software_fixture_cell(
        repository_root=ROOT, protocol=protocol,
        policy_id="multitrack-seed1", track_id="straight-sprint-v1",
        stratum="randomized_train", factory=_factory(),
        model_path=path, policy_for_seed=lambda seed: FailingPolicy(RuntimeError("synthetic crash")),
        episodes=3,
    )
    assert cell["status"] == "invalid_aborted"
    assert cell["valid_episode_count"] == 0
    assert cell["invalid_episode_index"] == 0
    assert cell["episodes"][0]["error_type"] == "RuntimeError"
    assert "synthetic crash" in cell["episodes"][0]["error"]


def test_research_cell_denied_before_model_or_policy_access(tmp_path):
    protocol = _protocol()
    called = []
    def should_not_load(seed):
        called.append(seed)
        raise AssertionError("policy accessed despite failed admission")
    with pytest.raises(ScientificAdmissionError):
        evaluate_registered_cell(
            ROOT, protocol, policy_id="multitrack-seed0",
            track_id="straight-sprint-v1", stratum="nominal",
            model_path=tmp_path / "nonexistent.zip", policy_for_seed=should_not_load,
        )
    assert called == []


@pytest.mark.parametrize("bad_seed", [-1, True, 2**32])
def test_legacy_evaluator_rejects_invalid_seed(bad_seed):
    from f1rl.learning.evaluation.evaluator import evaluate_episode
    with pytest.raises(ValueError, match="reset_seed"):
        evaluate_episode(
            track_id="fixture", factory=_factory(),
            policy=ConstantPolicy(np.array([0.0, 0.0], dtype=np.float32)),
            reset_seed=bad_seed,
        )


def test_cell_evidence_persists_checksum_and_refuses_overwrite(tmp_path):
    import json
    from f1rl.research.wp7_cell_evaluation import save_cell_evidence
    protocol = _protocol()
    model = tmp_path / "fixture.zip"
    model.write_bytes(b"synthetic model bytes")
    output = tmp_path / "cell.json"
    row = execute_software_fixture_cell(
        repository_root=ROOT, protocol=protocol,
        policy_id="multitrack-seed2", track_id="straight-sprint-v1",
        stratum="nominal", factory=_factory(max_steps=1), model_path=model,
        policy_for_seed=lambda seed: ConstantPolicy(np.array([0., 0.], dtype=np.float32)),
        episodes=2, evidence_path=output,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(row))
    with pytest.raises(ValueError, match="destination"):
        save_cell_evidence(row, output)
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(json.dumps(row))
