from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from src.evaluation import evaluate_episode
from src.evaluation.evaluator import _valid_terminal_failure
from tests.integration.test_hard_route_terminal_v2 import owner_factory
from tests.learning.evaluation.mocks.policies import ConstantPolicy


def _owner_failure_info() -> dict:
    return {
        "research_outcome": {
            "version": "wp7_episode_outcome_v1",
            "kind": "valid_terminal_failure",
            "valid_sample": True,
            "reason": "hard_route_violation",
        },
        "race": {"terminal_failure": True},
    }


def test_owner_approved_hard_route_failure_is_counted_as_valid_unsuccessful_episode():
    observed = []
    result = evaluate_episode(
        track_id="owner-hard-route",
        factory=owner_factory(),
        policy=ConstantPolicy(np.zeros(2, dtype=np.float32)),
        step_observer=observed.append,
    )
    assert result.success is False
    assert result.truncated is False
    assert result.steps == 2
    assert result.completion_time is None
    assert observed[-1]["terminated"] is True
    assert observed[-1]["truncated"] is False
    assert _valid_terminal_failure(observed[-1]["info"])


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("research_outcome", "version", "other_version"),
        ("research_outcome", "kind", "invalid_environment_transition"),
        ("research_outcome", "valid_sample", False),
        ("research_outcome", "valid_sample", 1),
        ("research_outcome", "reason", "collision"),
        ("race", "terminal_failure", False),
        ("race", "terminal_failure", 1),
    ],
)
def test_hard_route_terminal_failure_requires_unambiguous_registered_evidence(section, field, replacement):
    info = deepcopy(_owner_failure_info())
    info[section][field] = replacement
    assert not _valid_terminal_failure(info)


@pytest.mark.parametrize("missing", ["research_outcome", "race"])
def test_hard_route_terminal_failure_requires_both_evidence_sections(missing):
    info = _owner_failure_info()
    del info[missing]
    assert not _valid_terminal_failure(info)


def test_hard_route_terminal_failure_rejects_absent_or_non_mapping_evidence():
    assert not _valid_terminal_failure({})
    assert not _valid_terminal_failure(None)
    assert not _valid_terminal_failure({"research_outcome": None, "race": []})
