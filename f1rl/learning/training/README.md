# F1-RL training and vector environment contracts

The **same-track** vector environment uses `make_vector_env` with a single `F1RLEnvFactory`, while the **multi-track** option uses `make_multi_track_vector_env` with a registered `TrainingScenarioSet`, `track_id` and `sampling_seed`. The latter samples only declared scenarios, never automatically imports an unseen-track or HELD-OUT dataset.

Backends are `dummy` (in-process) and `subproc` (separate workers). For Windows-safe `subproc`, use the `spawn` start method; put the caller under `if __name__ == "__main__":` to avoid recursive process creation. A multi-track environment returns the selected `track_id` in each step's info dictionary. `sampling_seed` controls deterministic scenario choice per vector rank; canonical TRAIN reset seeds use the separate reset contract.

Use the configuration reference `configs/learning/vectorized-ppo.yml` only where it exists and has been independently reviewed. A diagnostic example is `python scripts/benchmark_vector_envs.py --backend dummy --n-envs 1 --transitions 4`. This diagnostic measures throughput but is not scientific training, physical validation or authorization.

Always call `close()` on vector environments in a `finally` block. Non-track environments, mismatched observation/action spaces, malformed seed metadata and unregistered scientific physics must be rejected instead of bypassing guards. The `training runner` has additional source and permission checks.
