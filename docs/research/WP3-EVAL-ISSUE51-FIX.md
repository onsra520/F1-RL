# Issue #51: WP3 evaluation reporter fix and execution constraints

Training evidence remains bound to the immutable source commit
`fe83f517479f3b990028d6409304fae83b8199a9` on
`research/wp3-ppo-tuning-v2`. This fix branch is a separate
**evaluation-source revision**, not a new training source. Do not
rewrite model manifests, original campaign provenance, seed rosters,
the 4 m evaluation boundary, or any of the existing model weights.

## Root cause and fix

At the end of `run_evaluation`, the code passed local names
`models` and `cells` to `_write_evaluation_reports`, although those
names were not bound. The correct already-built lists are
`model_records` and `all_cells`. The typo raised `NameError`
after the evaluation JSON may have been durably written, causing a
misleading failed exit and missing charts/reports. The patch only
corrects these arguments; it does not rerun PPO.

Run the regression test (and the remaining WP3 test suite) on a
machine with the project's Python dependencies installed:

```powershell
python -m pytest -q tests/research/test_wp3_evaluation_issue51.py tests/research/test_wp3_campaign.py
```

The existing `_write_new_or_match` writer refuses to replace a
different report and allows identical report content on restart.
Reopening evaluation evidence still requires careful handling:
`WP3-TRAIN-EVALUATION.json` is immutable by checksum and comparison;
the evaluator may not be used to overwrite different per-cell evidence.

## Current blocker and required provenance

The 12 models and telemetry are in
`handoff/wp3-khanh-bang-20261021` and the private prerelease.
The *original* campaign state is stale (2/12) and original external
smoke/preflight prerequisites at `D:\Projects\.evidence\F1-RL`
were not transferred. Do not fake those files, change their hashes,
or claim canonical campaign resume.

Actual registered TRAIN evaluation has **not** been performed as part
of this fix branch. To execute a complete evaluation, the separate
evaluator must independently revalidate the existing model hashes,
load the original frozen roster, preserve paired seeds and stochastic
policy predictions, record all 96 cells / 5,760 episode outcomes,
and record the new evaluation source commit separately from the
original training source. It must not read HELD-OUT data.

**Scientific WP7 and Gate 2 remain BLOCKED** until a separate
authorized process completes scientific acceptance.
