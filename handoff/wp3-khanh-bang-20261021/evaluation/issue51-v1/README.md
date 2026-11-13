# WP3 TRAIN evaluation — Issue #51

Outcome: PASS_WITH_INVALID_CELLS; 96/96 cells processed, 31 fully valid cells and 65 invalid-aborted cells. The run logged 2,153 valid episodes, 65 invalid episode records, and 3,542 remaining episode positions were never attempted after cell aborts.

There is no complete valid A/B/C/D ranking and no scientific acceptance.

TRAIN source (immutable): fe83f517479f3b990028d6409304fae83b8199a9

Evaluator execution source (separate bugfix PR #52): 4e6396a4000263ec610adaf8765f9ff8d899e6ca

Per-cell JSON summaries, append-only episode events, roster and lifecycle logs are in evaluation/. Six original reports/charts and runtime logs are included. EVAL-EVIDENCE-SHA256.json binds every published input file by SHA-256. All cell summaries, event-log prefix hashes, status counts and the 12 published final model hashes were checked before packaging.

Canonical campaign state remains unreconciled pending genuine historical preflight records. No missing evidence was synthesized, no training was repeated, and the frozen training source/weights were not altered. All HELD-OUT, WP7, Gate 2 and scientific acceptance paths remain BLOCKED.

Issue: https://github.com/onsra520/F1-RL/issues/51

Evaluator fix PR: https://github.com/onsra520/F1-RL/pull/52
