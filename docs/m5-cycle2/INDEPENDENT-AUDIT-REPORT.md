# M5 Cycle 2 Independent Evidence Audit

Date: 2026-11-11
Verdict: `PARTIAL` — execution artifacts pass integrity review; model selection and fresh-roster candidate audit are blocked by the failed candidate gate.

## Cycle 2 checkpoint report audit

An independently authored read-only verifier (`scripts/m5_cycle2_independent_audit.py`) loaded the registered paired evaluation plan, parent checkpoint evaluation, Cycle 2 checkpoint evaluation and checkpoint bytes. For the checkpoint report it verified:

- report canonical SHA-256, evaluation-plan SHA-256, checkpoint file SHA-256, `heldout_access=false`;
- all 85 cases unique and present on the registered roster, including exact environment/policy seed pairs;
- each raw JSONL trace SHA-256, row count and contiguous step sequence;
- applied steering/throttle/brake means, speed mean/max, success and terminal status, per-case step count, progress, and all five recomputed stochastic group totals;
- all 10 deterministic Slalom diagnostic cases are present and separately counted.

Result: `PASS` for both parent and fine-tuned report: 85/85 cases and trace hashes verified in each; fine-tuned report SHA-256 `222530e103cb2992084ee7e6e5171b656825bcc6c5b7061cb539a2a014155110`; parent report SHA-256 `6206d4c8bd55921a4b5cc2eb72eaada05e320fa9eee53081fd8106aa2498f210`. Machine-readable outputs: `training-attempt-v5\parent-selection-baseline\INDEPENDENT-AUDIT.json` and `training-attempt-v5\INDEPENDENT-AUDIT.json`.

The parent baseline report contained 85 complete cases and reports 15/15 on each of the four non-Slalom stochastic tracks and 12/15 Slalom. The Cycle 2 checkpoint preserved the four tracks but fell to 11/15 Slalom, so it was not eligible.

## Stage B prior screening artifact reconciliation

The separate M5 Stage B report was also reconciled with its artifacts: 400/400 episode summaries, summary self-hashes, raw trace SHA-256 values, normalized trajectory hashes, scorecard rows and report scorecard cells matched. `trace_sha256` in that report is the normalized trajectory digest (the normalizer excludes `case_id`); `raw_trace_sha256` is the hash of the exact JSONL file bytes. These are intentionally different hash domains. The CSV header ` fresh_finish` contains a leading space; values match after header trimming. No data correction or registration change was made.

Stage B evidence root: `D:\Projects\.archives\F1-RL\demo-completion\m5-multicandidate-screening-20261111-reprocheck\stage-b`. Stage B report file SHA-256: `cc2ad959903b43e4d82a11b2b6d26a17ace06b82c195d205678e9e0348e2e1ff`. Machine-readable reconciliation: `STAGEB-INDEPENDENT-AUDIT.json` in the Cycle 2 archive; the full artifact inventory remains in the screening run's `SHA256-MANIFEST.json`.

## Limits / blocked audit gate

- The 85-case roster was used for checkpoint selection and early-stop decisions, so it is not a fresh independent audit roster.
- No candidate met selection criteria; therefore no paired parent/winner fresh-roster audit was performed. This is recorded as **NOT RUN / BLOCKED**, not PASS.
- Deterministic Slalom 10/10 has identical trajectories in each checkpoint cell and is not evidence of robustness.
- No scientific signoff, model freeze, final acceptance, HELD-OUT access, M6+ or `DEMO_READY` status is claimed.
