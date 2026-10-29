"""The stable inference CLI stays read-only and preserves its result envelope."""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.interfaces import inference_cli


@dataclass(frozen=True)
class ResetConfig:
    seed: int


@dataclass(frozen=True)
class Factory:
    reset_config: ResetConfig

    def create(self):
        return SimpleNamespace(seed=self.reset_config.seed, step=lambda _action: None)


@pytest.mark.parametrize("valid_samples,expected_exit", [
    ([True, True, True], 0),
    ([True, False], 2),
])
def test_inference_adapter_is_read_only_seeded_and_keeps_output_contract(
    tmp_path, monkeypatch, capsys, valid_samples, expected_exit
):
    from src.exploratory import pilot
    from src.interfaces import runtime_selection
    from src.rl.algorithms import ppo
    from src.validation import wp7_protocol

    runtime_calls = []
    monkeypatch.setattr(runtime_selection, "ensure_runtime",
                        lambda *args: runtime_calls.append(args) or None)
    loaded = []

    class Model:
        def learn(self, *_args, **_kwargs):
            pytest.fail("inference called PPO training")

    monkeypatch.setattr(ppo.PPOModel, "load",
                        lambda *args, **kwargs: loaded.append((args, kwargs)) or Model())
    factory = Factory(ResetConfig(seed=0))
    monkeypatch.setattr(wp7_protocol, "load_registration", lambda *args, **kwargs: {})
    monkeypatch.setattr(wp7_protocol, "prepare_scenarios", lambda *args, **kwargs: SimpleNamespace(
        scenarios=[SimpleNamespace(track_id="straight-sprint-v1", factory=factory)]))

    observed = []

    def evaluate_episode(session, model, name, episode):
        assert name == "trained_final"
        env = session.create_env(learning=False, episode=episode)
        observed.append(env.seed)
        with pytest.raises(ValueError, match="inference only"):
            session.create_env(learning=True, episode=episode)
        return {"episode": episode, "valid_sample": valid_samples[episode]}

    monkeypatch.setattr(pilot, "evaluate_episode", evaluate_episode)
    thread_limits = []
    output = tmp_path / "inference.json"
    result = inference_cli.main(
        tmp_path,
        ["--model", "saved-policy.zip", "--episodes", str(len(valid_samples)), "--seed-start", "34",
         "--output", str(output)],
        torch_factory=lambda: SimpleNamespace(set_num_threads=thread_limits.append),
    )

    assert result == expected_exit
    assert len(runtime_calls) == 1
    assert loaded[0][1] == {"device": "cpu"}
    assert thread_limits == [1]
    assert observed == list(range(34, 34 + len(valid_samples)))
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document == {
        "training_started": False,
        "track": "straight-sprint-v1",
        "episodes": [{"episode": i, "valid_sample": is_valid}
                     for i, is_valid in enumerate(valid_samples)],
    }
    assert json.loads(capsys.readouterr().out) == document
