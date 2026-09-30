# `artifacts`

## Purpose

Generated outputs from training and evaluation.

## Subdirectories

- `checkpoints/` — intermediate training snapshots.
- `policies/` — exported policies selected for inference/demo.
- `evaluations/` — generated metrics, summaries, and experiment outputs.
- `runs/` ? per-run training state, checkpoints, and final training models.

Large generated files are ignored by default. Commit an artifact only when there is a deliberate reproducibility/demo reason.
