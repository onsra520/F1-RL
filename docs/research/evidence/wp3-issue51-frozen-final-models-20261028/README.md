# WP3 Issue #51 - Frozen 12 final PPO checkpoints

This artifact directory contains the original 12 final-only model-final.zip files (A/B/C/D x seed0/seed1/seed2) already used, unmodified, by owner-approved TRAIN-development evaluation FULL-06.

- Original training source commit: fe83f517479f3b990028d6409304fae83b8199a9 (not changed).
- Evaluation code source commit: 6c479540134d98d7d27331879c926c8ba82c2965 (separate from original training source).
- FULL-06 original owner-approved TRAIN-development result: 96/96 cells, 5,760 valid episodes, zero invalid, source-bound roster and 12 model hashes.
- All 12 model-final.zip files were copied from wp3-evidence/attempt-v8/runs/<model-id>/model-final.zip, verified against the exact 12 SHA-256 hashes in FULL-06 frozen RUN-MANIFEST.json, then independently rehashed after copying. ZIP CRC and contents were checked.
- Each weight archive is stored as models/<model-id>/model-final.zip. Exact SHA-256 hashes, byte sizes and archive entry names are in WP3-FROZEN-MODELS-SHA256-MANIFEST.json.
- No retraining, no checkpoint replacement, no original source/campaign alteration. Raw FULL-06 reports are separately stored in the wp3-issue51-full06-traindev-20261027 evidence directory.
- NOT a scientific release: independent Physics v4 signoff, scientifically registered evaluation, HELDOUT/WP7/Gate 2 are still blocked pending review.

Do not infer policy quality from checkpoint validity: the 12 models are integrity-verified, but the full run observed only 1,550 successful finishes out of 5,760 formally valid episode outcomes.
