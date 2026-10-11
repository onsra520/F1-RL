# WP7 Recovery — authorized exploratory real PPO run

User authorized starting a real 200k pilot on 2026-10-11. Scientific preflight still says `review_required`, so this is **exploratory** calibration, **not** a frozen experiment and not evidence of a verified 4/4 generalization claim.

## Source boundary / exact command

From `recovery/wp7-rebuild`, after tests pass and Git is committed/clean:

```powershell
.\.venv\Scripts\python.exe scripts\wp7_run_exploratory.py --role calibration --accept-review-risks --artifacts-root "D:\Projects\.archives\F1-RL\wp7-exploratory"
```

The full command requires `--accept-review-risks`, exposes preflight warnings, and only constructs four TRAIN scenarios. The CLI also permits `--role confirmation` with independent PPO/sampling seed `1/1`, but **does not start confirmation automatically**. Both variants retain 200k requested steps, eight DummyVecEnv CPU, MlpPolicy, checkpoints every 25k, and one deterministic TRAIN-only episode per track per saved archive.

`PilotMilestoneRecorder(exploratory_mode=True)` allows evaluating real model archives while a scientific preflight is still warning, but **still requires a clean Git revision**, checks source fingerprints repeatedly, validates that evaluation loads real `PPOModel.load`, records nominal and actual steps and report/model SHA256, and refuses any held-out track. All reports say `scientific_evidence: false` and `status: exploratory_unreviewed_not_scientific`. The old strict default remains review-blocked, and the separate mock-only mode remains unchanged.

Artifacts are directed outside the worktree under `D:\Projects\.archives`, so branch cleanup should not remove checkpoint bytes. This is **the same computer/drive**; it is NOT an off-device disaster-proof backup. Do not commit training weights to Git without an explicit storage policy.

## Preconditions verified

- Fresh real smoke run: 2,048/2,048 PPO timesteps on 8 DummyVecEnv CPU, model ZIP saved successfully.
- Real smoke archive reloaded with `PPOModel.load(device="cpu")`; deterministic checkpoint evaluator ran only on all four TRAIN tracks and verified its SHA256. The early smoke policy achieved **0/4**; this is a *diagnostic*, not a pilot result.
- Route slalom navigation already repaired. Remaining preflight blockers: swept vehicle footprint/collision clearance, 90-degree right-turn geometry, and scientific protocol/provenance approval.
- Before full run: regression suite pass, committed clean Git and explicit user authorization.

## Monitoring

Output is JSONL with `training_start`, `milestone_verified`, `final_milestone_verified`, `postrun_checksum_verification`, `training_complete`. `state.json` inside the run directory is authoritative for `running`, `failed` or `completed` and actual timesteps.

**No final scientific seed replicates, learned-policy HELD-OUT evaluation, WP8 or merge into main.**
