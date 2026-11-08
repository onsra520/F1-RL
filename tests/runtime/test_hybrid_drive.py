from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from f1rl.local_demo.runtime.api import CreateSessionRequest
from f1rl.local_demo.runtime.hybrid_drive import HybridDrive


def _hybrid(ppo_action, auto_action, *, weight=0.5, tick=0):
    env = SimpleNamespace(runtime_snapshot=lambda: {"tick": tick, "dt_s": 0.1})
    ppo = SimpleNamespace(predict=lambda _observation, deterministic: np.asarray(ppo_action, dtype=np.float32))
    auto = SimpleNamespace(
        env=env,
        predict=lambda _observation, deterministic: np.asarray(auto_action, dtype=np.float32),
        requires_obstacle_assist=lambda _snapshot: False,
    )
    return HybridDrive(ppo, auto, weight)


def test_hybrid_blends_steering_and_limits_throttle_to_both_controllers():
    controller = _hybrid([0.4, 0.8, 0.0], [-0.2, 0.6, 0.0])

    np.testing.assert_allclose(controller.predict(None), [0.1, 0.6, 0.0], atol=1e-6)
    assert controller.status()["steps"] == 1


def test_hybrid_obstacle_takeover_uses_auto_steering():
    controller = _hybrid([1.0, 0.4, 0.0], [-0.2, 0.7, 0.0])
    controller.auto.requires_obstacle_assist = lambda _snapshot: True

    np.testing.assert_allclose(controller.predict(None), [-0.2, 0.4, 0.0], atol=1e-6)
    assert controller.status()["obstacle_takeover_steps"] == 1


def test_hybrid_auto_braking_overrides_ppo_throttle():
    controller = _hybrid([1.0, 1.0, 0.0], [-1.0, 1.0, 0.4], weight=0.2)

    np.testing.assert_allclose(controller.predict(None), [-0.25, 0.0, 0.4], atol=1e-6)


def test_hybrid_rejects_invalid_actions():
    controller = _hybrid([float("nan"), 0.5, 0.0], [0.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="invalid action"):
        controller.predict(None)


def test_hybrid_session_contract_requires_exploratory_model_and_default_physics():
    request = CreateSessionRequest(
        mode="hybrid", model_id="wp7-calibration", track_id="right-turn-v1", seed=0,
    )
    assert request.mode == "hybrid"

    with pytest.raises(ValidationError):
        CreateSessionRequest(
            mode="hybrid", model_id="A-seed0", track_id="right-turn-v1", seed=0,
        )

    with pytest.raises(ValidationError):
        CreateSessionRequest(
            mode="hybrid", model_id="wp7-calibration", track_id="right-turn-v1", seed=0,
            physics_profile="experimental_v4",
        )
