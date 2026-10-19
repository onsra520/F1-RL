# Demo-ready local exploratory training

**Use the isolated demo branch only.** From the checked-out, clean branch,
double-click **`START-DEMO-TRAIN.cmd`** in the repository root. It chooses
`F1RL_PYTHON`, the local `.venv`, Python 3.12 launcher or system Python.

You will see the resolved scope and must type **TRAIN** to create a fresh
one-use local consent. No Ed25519 private key is required for this *local
exploratory-only* path. If canceled, no Gym rollout/model training starts.
No scientific admission, HELD-OUT, WP8, PR merge or cloud compute is permitted.

The script prepares and validates a fresh bound permit, creates a unique
**external evidence** folder, then starts the existing supervisor. It displays
`state.json` progress and the evidence directory. Do **not** rerun a consumed
permit or use a prior `run/` folder; each click receives a fresh identity.

## Prerequisites

Python 3.11+, PyTorch, Gymnasium, Stable-Baselines3, PyYAML and cryptography
must exist in the selected interpreter. In the repository checkout:

```powershell
python -m pip install -e ".[rl,science]"
```

If the project already has a separate tested interpreter, specify it once:

```powershell
$env:F1RL_PYTHON = "D:\path\to\working\.venv\Scripts\python.exe"
.\START-DEMO-TRAIN.cmd
```

The script does not download/install unapproved dependencies by itself.

## Current demo budget

The current `config.yml` intentionally sets **4,096 training interactions,
512-step checkpoints, two evaluation episodes per policy**, CPU
`threads: auto`, 1 environment, and a hard total wall budget of 3,600 s.
This is a small **exploratory demo**, **not** proof of reliable autonomous
course completion. You can increase the YAML steps (within bounded safeguards)
for a later, owner-approved run. GPU is disabled.

`learning_first`:

1. Fresh scoped authorization, create PPO, save an immutable `untrained.zip`.
2. **PPO learns first**, saves periodic checkpoints.
3. Write `trained-final.zip` and `learning-summary.json` immediately after
   real optimizer updates and changed weights are verified.
4. Evaluate trained first, then random, then **reload untrained.zip** to
   evaluate the original weights on the same fixed evaluation seeds.
5. Replay the selected trained episode and produce `comparison.json` only if
   all mandatory evaluations and replay succeed.

If the 60-min worker deadline expires, finished checkpoint ZIPs, artifact
ledger and `learning-summary.json` (if reached) remain available. A completed
learning phase with incomplete evaluation is **not** reported as a completed
experiment. Never claim autonomous success solely from a changed weight hash.

## Local verification (no PPO training)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -B -m pytest -q tests/research/test_exploratory_config.py tests/research/test_exploratory_permit.py tests/research/test_exploratory_pilot.py tests/learning/training/test_run_config.py
```

Then run `START-DEMO-TRAIN.cmd` manually when you are prepared for the compute.
