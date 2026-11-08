from pathlib import Path

from f1rl.local_demo.runtime.catalog import RuntimeCatalog

ROOT = Path(__file__).resolve().parents[2]


def test_b_seed1_slalom_seed_412_remains_grounded_through_86_train_steps():
    catalog = RuntimeCatalog(ROOT)
    env = catalog.create_environment("slalom-v1", seed=412, mode="manual")
    try:
        observation, _ = env.reset(seed=412)
        model = catalog.load_model("B-seed1", env=env)

        for _ in range(86):
            action = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
        assert info["tick"] == 86
        assert info["telemetry"]["grounded"] is True
    finally:
        env.close()
