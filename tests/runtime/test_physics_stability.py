from pathlib import Path

import pytest

from src.runtime.catalog import RuntimeCatalog
from src.simulation.independent_suspension_dynamics import SuspensionResolutionError

ROOT = Path(__file__).resolve().parents[2]


def test_b_seed1_slalom_seed_412_accepts_supported_overhang_then_classifies_runoff():
    catalog = RuntimeCatalog(ROOT)
    env = catalog.create_environment("slalom-v1", seed=412)
    try:
        observation, _ = env.reset(seed=412)
        model = catalog.load_model("B-seed1", env=env)

        for _ in range(85):
            action = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated

        action = model.predict(observation, deterministic=True)
        with pytest.raises(
            SuspensionResolutionError,
            match=r"^track_boundary_exit: front_right wheel center left road support$",
        ) as error:
            env.step(action)
        assert error.value.failure_code == "track_boundary_exit"
    finally:
        env.close()
