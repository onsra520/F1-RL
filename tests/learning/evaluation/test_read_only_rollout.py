from types import SimpleNamespace

import numpy as np

from src.evaluation.rollout import evaluate_policy_episode


def test_read_only_rollout_preserves_trace_and_reports_progress_without_step_patch():
    class Env:
        action_space = SimpleNamespace(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
        )
        _authored_route = SimpleNamespace(total_meters=100.0)

        def __init__(self):
            self.closed = False

        def reset(self, seed):
            self.seed = seed
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            assert action.tolist() == [0.25, 0.5, 0.0]
            return (
                np.array([1.0], dtype=np.float32),
                2.0,
                True,
                False,
                {
                    "research_outcome": {
                        "kind": "successful_finish",
                        "valid_sample": True,
                    },
                    "race": {
                        "completed_checkpoint_ids": ("finish",),
                        "directed_crossings": ("finish",),
                    },
                    "telemetry": {"obstacle_hit_ids": ()},
                    "route_highwater_meters": 100.0,
                    "elapsed_time": 0.02,
                },
            )

        def close(self):
            self.closed = True

    class Policy:
        def predict(self, observation, *, deterministic):
            assert deterministic is True
            return np.array([0.25, 0.5, 0.0], dtype=np.float32)

    env = Env()
    observed = []
    row = evaluate_policy_episode(
        env,
        Policy(),
        policy_name="trained_final",
        episode=0,
        environment_seed=45,
        on_step=lambda step, transition: observed.append(
            (step, transition[2], transition[3])
        ),
    )

    assert env.closed is True
    assert env.seed == 45
    assert observed == [(1, True, False)]
    assert row["policy"] == "trained_final"
    assert row["episode"] == 0
    assert row["environment_seed"] == 45
    assert row["valid_sample"] is True
    assert row["success"] is True
    assert row["progress"] == 1.0
    assert row["completed_checkpoint_ids"] == ["finish"]
    assert row["directed_crossings"] == ["finish"]
    assert row["collisions"] == 0
    assert row["off_route_events"] == 0
    assert row["duration_s"] == 0.02
    assert row["steps"] == 1
    assert row["return"] == 2.0
    assert row["outcome"] == {"kind": "successful_finish", "valid_sample": True}
    assert row["error"] is None
    assert len(row["trace_sha256"]) == 64


def test_read_only_rollout_records_invalid_transition_and_closes_env():
    class Env:
        _authored_route = SimpleNamespace(total_meters=100.0)

        def reset(self, seed):
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            raise RuntimeError("simulator failed")

        def close(self):
            self.closed = True

    env = Env()
    env.closed = False
    env.last_research_outcome = {
        "kind": "invalid_environment_transition",
        "valid_sample": False,
    }
    row = evaluate_policy_episode(
        env,
        SimpleNamespace(predict=lambda *_args, **_kwargs: np.array([0.0])),
        policy_name="trained_final",
        episode=2,
        environment_seed=9,
    )

    assert env.closed is True
    assert row["valid_sample"] is False
    assert row["error"] == "RuntimeError: simulator failed"
    assert row["outcome"] == env.last_research_outcome
