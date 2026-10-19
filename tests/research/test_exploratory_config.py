"""Configuration contract tests; no Gym reset, evaluation or PPO training."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from f1rl.exploratory.config import resolve
from f1rl.learning.training.config import TrainingRunConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config.yml"


def _write(tmp_path, change):
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    change(document)
    (tmp_path / "config.yml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return tmp_path


def test_root_v2_still_loads_old_general_training_contract():
    base = TrainingRunConfig.from_yaml(CONFIG)
    assert base.total_timesteps == 50000
    assert base.vector_env.n_envs == 8


def test_exploratory_permit_settings_come_from_one_yaml():
    current = resolve(ROOT)
    assert current["limits"]["learning_timesteps"] == 32768
    assert current["limits"]["checkpoint_steps"] == 8192
    assert current["seeds"]["training_reset"] == 1000
    assert current["seeds"]["evaluation"] == list(range(10000, 10032))
    assert current["limits"]["n_envs"] == 1
    assert current["limits"]["device"] == "cpu"


def test_auto_and_explicit_cpu_resources(tmp_path, monkeypatch):
    monkeypatch.setattr("f1rl.exploratory.config.os.cpu_count", lambda: 12)
    auto = resolve(_write(tmp_path, lambda data: None))
    assert (auto["limits"]["threads"], auto["limits"]["blas_threads"]) == (12, 12)
    def limited(data):
        data["exploratory"]["resources"]["cpu"]["threads"] = 8
        data["exploratory"]["resources"]["cpu"]["blas_threads"] = 4
    limited_result = resolve(_write(tmp_path, limited))
    assert limited_result["limits"]["threads"] == 8
    assert limited_result["limits"]["blas_threads"] == 4
    assert limited_result["limits"]["interop_threads"] == 1


def test_editing_yaml_changes_resolved_permit_binding(tmp_path):
    first = resolve(_write(tmp_path, lambda d: None))
    def different(d):
        d["exploratory"]["training"]["total_timesteps"] = 16384
        d["exploratory"]["evaluation"]["episodes_per_policy"] = 4
        d["exploratory"]["evaluation"]["seed_start"] = 12000
        d["exploratory"]["execution"]["wall_seconds"] = 600
        d["exploratory"]["logging"]["progress_every_steps"] = 128
    second = resolve(_write(tmp_path, different))
    assert first["config_sha256"] != second["config_sha256"]
    assert second["limits"]["learning_timesteps"] == 16384
    assert second["limits"]["wall_seconds"] == 600
    assert second["seeds"]["evaluation"] == [12000, 12001, 12002, 12003]
    assert second["logging"]["progress_every_steps"] == 128


@pytest.mark.parametrize("section,key,value", [
    ("training", "total_timesteps", 50000),
    ("training", "checkpoint_every_steps", 7),
    ("evaluation", "episodes_per_policy", 33),
    ("evaluation", "seed_start", True),
    ("evaluation", "deterministic", False),
    ("execution", "wall_seconds", 7200),
    ("execution", "schedule", "learning_first"),
    ("logging", "progress_every_steps", 17),
    ("logging", "save_metrics", False),
    ("vector_env", "n_envs", 8),
    ("experiment", "track", "held-out"),
    ("experiment", "physics_profile", "unknown"),
])
def test_unsafe_or_unsupported_scope_denied(tmp_path, section, key, value):
    def mutate(data):
        data["exploratory"][section][key] = value
    with pytest.raises(ValueError):
        resolve(_write(tmp_path, mutate))


def test_unavailable_cpu_budget_denied(tmp_path, monkeypatch):
    monkeypatch.setattr("f1rl.exploratory.config.os.cpu_count", lambda: 8)
    with pytest.raises(ValueError):
        resolve(_write(tmp_path, lambda d: d["exploratory"]["resources"]["cpu"].update(threads=16)))


def test_unknown_sections_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        resolve(_write(tmp_path, lambda d: d["exploratory"].update(authorize_scientific=True)))


def test_duplicate_yaml_keys_fail_closed(tmp_path):
    contents = CONFIG.read_text(encoding="utf-8")
    (tmp_path / "config.yml").write_text(contents + "\nppo: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        resolve(tmp_path)
