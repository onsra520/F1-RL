# WP3 Issue #51 — Source-lock policy (Physics v4 + Crash Terminal v1)

Work branch: `fix/wp3-eval-physical-completion` from handoff `ce0e55d7a428826b2d4ed3218641105bd0206f4b`.

## Scope of a locked source commit

Commit the modified production Physics, Simulation, Gym, evaluator and race modules; every dependent Python module and adversarial test; minimum reproducible model/seed evidence receipts and the replay/preflight scripts that can verify them. Keep the original frozen 12 PPO ZIPs, historical Eval JSONL/cells, independent review worktrees and HELD-OUT untouched.

Exclude iterative scratch-probe scripts and stepwise intermediate JSON diagnostics from the main source tree; preserve them in an external backup/quarantine with their original relative paths and file SHA-256 (no deletion).

## Historic evidence

`WP3-PHYSICS-V4-TECHNICAL-ACCEPTANCE-20261024.json` and `WP3-CRASH-TERMINAL-V1-TECHNICAL-REGISTRY-20261024.json` are **historic, source-bound engineering receipts authored before any source commit**. Their `source_base_commit`/`original_handoff_git_head` and `source_tree_clean=false` MUST stay as originally recorded; they are not evidence of a clean new HEAD. After source lock, the archive validators require the current HEAD to descend from the original handoff AND all referenced source, model, roster and pytest receipt hashes to match.

The clean source is established by a **separate** post-lock audit recording `git rev-parse HEAD`, `git status --porcelain`, the commit tree object, the complete tracked source file digest manifest, pytest transcript hash, and the frozen checkpoint/roster SHA. Never rewrite the old JSON receipts to pretend they were post-commit signoffs.

## Acceptance

Run full `pytest -q` from the clean locked worktree. The WP2 experiment manifest has an explicit clean-tree precondition; do not bypass it. If any test fails, fix and commit again, then run the full suite on the new clean HEAD.

Tag the passing commit and verify its object hash after the green run. The tag is a local source-freeze identifier, **not a scientific reviewer signature**. A separate independent physics-domain approval is still required before registered 5,760-episode Evaluation. No source-lock command may run TRAIN or HELD-OUT.
