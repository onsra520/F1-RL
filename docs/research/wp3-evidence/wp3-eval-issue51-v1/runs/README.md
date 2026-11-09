# Linked training-run artifacts are not included

The historical source `runs/` entries are **12 Windows junctions** to
a different `attempt-v8/runs/<model-id>` tree. This Git snapshot contains
the 202 regular source files, not those junction targets. There are
300 linked files (about 3.29 GB), including model ZIP archives.
The ZIP models/checkpoints must be archived separately with their
own hashes and explicit artifact provenance. Do not assume the Git
snapshot is a complete reproducible campaign.

| Model | External linked file count | External bytes (not uploaded) |
|---|---:|---:|
| A-seed0 | 25 | 274893052 |
| A-seed1 | 25 | 273991179 |
| A-seed2 | 25 | 273982965 |
| B-seed0 | 25 | 275120110 |
| B-seed1 | 25 | 273891354 |
| B-seed2 | 25 | 274553110 |
| C-seed0 | 25 | 274364082 |
| C-seed1 | 25 | 273932500 |
| C-seed2 | 25 | 274453428 |
| D-seed0 | 25 | 275154441 |
| D-seed1 | 25 | 275057498 |
| D-seed2 | 25 | 274501416 |
