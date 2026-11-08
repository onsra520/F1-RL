# F1-RL

Interactive reinforcement learning environment for autonomous racing and obstacle avoidance.

## Canonical source

- `f1rl/` owns Physics v4, vehicle and track dynamics, collision and race contracts, learning, PPO, and the development runtime.
- `web/` contains the Racing Sandbox editor, Track Builder, runtime controls, and replay viewer.
- `models/` contains the registered frozen PPO archives used for inference. Research admission remains fail-closed; this project state does not authorize training or scientific evaluation.

## Local runtime

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe --editable ".[dev,rl]" -r requirements-runtime.txt -r requirements-test.txt
cd web
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe run.py
```

The backend binds to `127.0.0.1:8000` and the frontend to `127.0.0.1:5173`. Use `run.py --backend-only` or `run.py --frontend-only` to start one side.

## Consolidation record

The final source provenance, archived inputs, migration decisions, and clean-build/regression results are in [CONSOLIDATION-REPORT.md](CONSOLIDATION-REPORT.md).
