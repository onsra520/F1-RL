# Final consolidation audit

Historical checkpoint verdict at `1e079f6`: **BLOCKED**. See the remediation follow-up below for the current status; this historical section records the earlier state. Baseline `69acffa0accee5806923db22e40c029ff82007c5`. No merge or cleanup is authorized by this report.

## Provenance and decisions

- Source and per-file SHA-256, size, original path, destination and purpose for the bundled files: `FINAL-CONSOLIDATION-ASSETS.json`. The 12 final weights and their manifest originate at `879c124e33a3c4ea65952ede11cc202c3fb1b3f6`; FULL-06 original bytes were restored from `evidence-wp3-87face7-raw.zip` for source `87face7416cbf9fa31cbbc76448cb0fe70774b75`. Git blobs for reports and JSONL were line-ending normalized and do not match the original manifest. The archive's original bytes do; `.gitattributes` preserves them on checkout. No scientific IDs, manifest values, model ZIPs or JSONL contents were edited.
- `FINAL-CONSOLIDATION-HANDOFF-INVENTORY.json` enumerates all 501 archive members. `handoff-wp3-2ae51b6-manifest-verified.zip` remains outside the branch at the paths recorded in that inventory. Its 501 files passed ZIP CRC; 12 model files were independently byte-hash matched to the bundled weights. The historical stdout log reconstruction exception and the separately retained raw Git archive are documented in `PRE-CLEANUP-CONSOLIDATION.md`. No historical member was deleted or claimed as a duplicate merely by name.
- The baseline incorporates the frontend branch and Physics v4 roll-envelope commit as Git ancestors, and prior consolidation documents WP1E/WP7 ports and regressions. The quaternion roll alternative `13b1d8105d14e7492a140a850c43f22e578fd809` has different semantics and is retained in the verified bundle, not enabled. This is not a complete patch-by-patch audit of every historical branch; see `PRE-CLEANUP-CONSOLIDATION.md` for prior decisions.
- The 12 models and FULL-06 output are now in the branch. Existing scientific admission remains fail-closed. UI to Python runtime bridge: **NOT IMPLEMENTED**.
- Historical admission still requires the roster, provenance and telemetry for 12 runs. The existing reconstructed `runs` directory is 3,293,895,135 bytes and is outside this checkout, with junctions to a temporary location. `F1RL_WP3_EVIDENCE` now overrides the old Windows default for the relevant production entry points, but a clean checkout alone cannot satisfy this historical preflight. Do not treat the temporary fixture as a durable backup. Full history recovery and any safe, source-verified reduction of the telemetry dependency remain open.

## Integrity and recovery

`python scripts/verify_consolidated_wp3_assets.py` exit 0: 12/12 model hashes and ZIP CRCs, canonical run manifest and output-manifest hash, 7/7 reports and 96/96 cell hashes. Independent parsing counted 96 cells, 5,760 started and completed episodes and 1,550 finishes, with outcome counts matching the receipt. These checks are not an independent scientific validation of the episodes. A copied checkpoint deliberately corrupted under `python -O` was rejected by the verifier.

Archive root: `D:/Projects/.archives/F1-RL/pre-cleanup-consolidation-818eedc/`.

- `evidence-wp3-87face7-raw.zip`: SHA-256 `144a41a5e2b2f393103321487265a6ba0f98ec1075196e4f796aecf72b3ef2ba`, 209 files, ZIP CRC passed; 113 FULL-06 files restored into this branch and original report/cell checksums matched.
- `handoff-wp3-2ae51b6-manifest-verified.zip`: SHA-256 `585eabeba9003dfc5db389e53051c6a9d0c71d0a7b171a5ca2bdbaf95ec8a034`, 501 files, ZIP CRC passed; per-member hashes recorded in inventory. All 501 members were extracted to an independent temporary directory and matched against the per-file inventory. Original handoff, continuation and evaluation manifests were independently checked: 73/73, 216/216 and 204/204 entries matched.
- `unique-branches.bundle`: SHA-256 `58387b8755ed496ecb48630ad9abda15915794901a8aac624831678060b0a22b`; `git bundle verify` exit 0, complete history for seven listed refs. This bundle contains `87face7`, not the later frozen-model commit `879c124`; the 12 model bytes are also retained in handoff archive and this branch. An independent clone, explicit fetch of all seven refs into `refs/recovered/*`, and `git fsck --full` succeeded; all seven recovered commit IDs were inspected, including the quaternion alternative.

## Verification matrix

Set `PYTHONPATH` to the checkout for subprocess-based tests. All checks below were run in this consolidation worktree, with the changes present, unless explicitly stated otherwise. External fixture `F1RL_WP3_EVIDENCE=D:/Projects/.evidence/F1-RL/WP3-EVIDENCE/wp3-eval-issue51-v1`; Python environment `D:/Projects/.temp/F1-RL-wp7-scientific-venv/Scripts/python.exe`. CPU library thread counts were set to 2 for Python regression. No OS CPU affinity limit.

| Check | Result and rerun |
| --- | --- |
| Integrity | PASS, exit 0: `python scripts/verify_consolidated_wp3_assets.py` |
| WP3 admission and path | PASS, exit 0: 80 passed; `python -m pytest -q tests/research/test_wp3_evidence_path_config.py tests/research/test_wp3_eval_only_admission.py tests/research/test_wp3_v4_technical_acceptance.py tests/research/test_wp3_current_eval_gate.py tests/research/test_wp3_crash_terminal_registry.py tests/research/test_wp3_issue51_latest_gate.py` (set fixture env above). |
| Frontend | PASS: `cd web; npm test; npm run typecheck; npm run build`, each exit 0; 12 tests. Build warns of a chunk >500 kB. |
| Runtime | PASS as smoke only: real bundled A-seed0 loaded with PPO (65,536 training steps); `python scripts/smoke_consolidated_wp3_model.py` on TRAIN straight-sprint, nominal, environment seed 123 and policy seed 456, ended truncated at 500 steps, failure reason `step_limit`, zero checkpoints. No HELD-OUT, training or scientific approval. |
| Full Python | INCOMPLETE, no pytest summary. First attempt exited 2 at collection because the environment lacked `cryptography`; it was installed. A second attempt lacked `PYTHONPATH`, timed out at 300 s after 69% (2,117 success, 9 failure, 3 error and 2 skip progress markers); a separate `pytest -xq` identified the first failure as a subprocess unable to import `f1rl`. This attempt overlapped code edits, so is not a final-tree result. After setting `PYTHONPATH` to the checkout, `pytest -q tests/integration/test_vectorized_ppo_integration.py` passed 12/12. The final configured `python -m pytest -q` on commit `f416a57` timed out after 300 s at 69%: 2,102 success and 2 skip progress markers, 0 failure/error markers, no summary. These counts are progress markers, not a pytest result. Logs: `full-python-final-consolidation.log` and `full-python-configured-consolidation.log` in the archive root. |
| Clean checkout | PARTIAL PASS: local no-hardlinks clone of commit `f416a57`, `python scripts/verify_consolidated_wp3_assets.py`, imports of `f1rl`, Gym and the WP3 gate, `cd web; npm ci; npm run build`, all exit 0. The imported package path was the independent clone. `npm ci` reported 6 vulnerabilities (3 moderate, 1 high, 2 critical) and blocked three install scripts; build nevertheless succeeded. Historical preflight still depends on the external fixture. |
| Full safety, lifecycle, physics, Gym, PPO and WP7 | Not independently established by a final summarized full run; do not infer PASS from the prior baseline report. |

## Remaining gates at the historical checkpoint

Complete the full branch-history and semantic audit, establish portable source-verified historical admission inputs without weakening the gate, complete source-level historical audit and independent portability of admission inputs, run and diagnose the complete Python regression with a final summary, and run runtime/terminal checks sufficient to cover all required outcomes. Only propose a fast-forward once every mandatory gate passes. Independent Physics v4 scientific review and registered evaluation remain pending; FULL-06 is TRAIN-development evidence only.

# Remediation follow-up

Engineering verdict: **CONDITIONAL**. All grouped tests have final summaries and the restore path is demonstrated, but the groups were run across two successive commits rather than on one exact source tree and the broad semantic history review remains limited. Do not propose fast-forward on this evidence alone. Scientific registered evaluation and independent physics approval remain BLOCKED; no scientific permission is granted.

## Reproducible historical bootstrap

Checkout contains the 12 frozen weights and 26 exact historical roster/provenance/validation/state files, with per-file provenance in `FINAL-CONSOLIDATION-ASSETS.json`. Large raw telemetry is not in Git. Download these immutable assets from the GitHub release `wp3-khanh-bang-20261021`:

- `WP3-attempt-v8-full-evidence.zip` — SHA-256 `fd39193a8689db1ba529910851435a46402981372b1dd893c7fd523b72363edf`, [release asset](https://github.com/onsra520/F1-RL/releases/download/wp3-khanh-bang-20261021/WP3-attempt-v8-full-evidence.zip).
- `WP3-attempt-v8-continuation-9-telemetry-20261022.zip` — SHA-256 `726573b374f3aec727457f2c7b7b1f6533ae09079e257577b85ced17fac18926`, [release asset](https://github.com/onsra520/F1-RL/releases/download/wp3-khanh-bang-20261021/WP3-attempt-v8-continuation-9-telemetry-20261022.zip).

With Python and the project dependencies installed, run from a full Git clone (the preflight checks source ancestry):

```powershell
python scripts/restore_wp3_historical_evidence.py --full <full-zip-path> --continuation <continuation-zip-path> --destination <new-history-directory>
$env:F1RL_WP3_EVIDENCE = '<new-history-directory>'
```

Use a destination that does not yet exist and has at least 3.3 GB free, plus temporary extraction overhead. The bootstrap verifies archive SHA-256, pins the 26-file inventory, verifies those byte hashes and the frozen models/FULL-06 manifests, streams only required members, then invokes the unchanged historical preflight including 12 model, telemetry and engineering-validation hashes, roster and state identity. It expects the scientific gate to remain blocked. An interrupted final publication can leave a partial destination: inspect and remove that destination before retrying; the tool never replaces an existing destination.

Source commits are 2ae51b61d6ee84bf8cbfb58013b2c92f87031df1 for the 26 inputs, and 879c124e33a3c4ea65952ede11cc202c3fb1b3f6 for the frozen models. All 26 input bytes were directly compared to their 2ae51b6 Git blobs. Independent checkout `616337d` reconstructed from the two ZIPs, verified 12 models and telemetry and original roster `f2f9d15954cea48345a8932773ed945f6ed250ad1435b4e675b97e4a6b94f319`, and returned `scientific_gate: BLOCKED`, zero new episodes. Historical input provenance SHA-256 `3b0b9c2bcdaf7614b7a1164b78a9ca0441bb8986488b8249c8614e1a63085349`; roster file SHA-256 `1530a54a636f80cacc096f2ed86d9e89fd5e7c7a7429873deb2fe366bcd859dc`. `pytest -q tests/research/test_wp3_historical_restore.py`: 3 passed, including corrupt inputs and existing-destination checks. Tampered input does not publish a destination.

A full checkout is self-contained for source, weights and FULL-06 verification; historical preflight requires the two separately distributed immutable release assets. No research worktree or old branch checkout is required. The release files remain external availability dependencies. `pip install --no-deps --no-build-isolation --target <directory> .` built a wheel and isolated import used that target; independently cloned `8893eec` also ran real bundled PPO inference for 500 TRAIN-development steps, ending `step_limit`. Full engineering pass cannot be inferred from this smoke.

## Branch and contract decisions

`git fetch origin --prune`, `git merge-base --is-ancestor`, and `git cherry HEAD <ref>` compared local and remote refs. Frontend racing sandbox, main integration, WP3 PPO tuning, physical-completion/road-impact/roll-envelope, WP7 recovery and all other local WP7 refs are **INCLUDED** ancestors. WP1E recovery/hardening are non-ancestors and were compared by commit and behavior (see `PRE-CLEANUP-CONSOLIDATION.md`): draft permit signatures `1af15fd`/`592be32`, cleanup/finalization `d8d0a03`/`5f1f217`/`4d4348a`, capacity `97f7ffa`/`3510c20`, real reset `af79543`/`a166742`/`b5e3969` are **INCLUDED or EQUIVALENT** through consolidation's independently adapted patches and regressions; draft curriculum documentation `1e50694`/`ecfdbea` is **INCLUDED**. The more conservative rollout-capacity rule is an explicit compatibility restriction. Independent suspension `be63b0b` regression is **INCLUDED**, its predecessor source change is **EQUIVALENT**. Quaternion `13b1d81` is **ARCHIVED / REJECTED from active physics** because it crosses a roll boundary that the active `fc18787` roll-envelope contract makes absorbing; it is retained in the verified bundle, not semantically equivalent.

Handoff commits `97ac293`, `306d428`, `e6a5758`, `2ae51b6` are **ARCHIVED**, with 12 models and required small preflight inputs **INCLUDED**; archives and 501-member inventory provide recovery. Fifteen evidence-only commits from `255bf39` through `879c124` have no source files; frozen models and canonical FULL-06 are **INCLUDED**, other evidence is **ARCHIVED** in `evidence-wp3-87face7-raw.zip` and the verified bundle. The later model commit is preserved by the branch and the independently verified handoff archive. Archify `ef95ec5` is **INCLUDED** as a warned historical snapshot. No claim is made that patch identity alone proves physics equivalence. Additional research branch semantics beyond these recorded source changes remain a review limitation.

## Remediation test matrix

All groups below had exit 0 and final pytest summaries. Windows used Python 3.12, `PYTHONPATH` set to the active checkout, `F1RL_WP3_EVIDENCE` set to the independently restored fixture, and OMP/MKL/OpenBLAS/NumExpr thread limits of 2; no OS CPU affinity limit. Research and sandbox file partitions use the sorted file list, slices `files[i::4]` or `files[i::3]` respectively. The two slow research groups ran concurrently with another group; at most three test processes were active. Logs are in `D:/Projects/.archives/F1-RL/pre-cleanup-consolidation-818eedc/grouped-8893eec/`, `independent-8893eec/`, and `split-616337d/`. The complete file inventory was compared to the partition: 254/254 `test_*.py` files accounted for.

| Group | Outcome | Source tree |
| --- | --- | --- |
| Research 1 | 244 passed | `8893eec` |
| Research 2, fast | 177 passed | `616337d` |
| Research 2, coupled ramp | 1 passed | `616337d` |
| Research 2, rotating owner | 4 passed | `616337d` |
| Research 2, spatial profile | 19 passed | `616337d` |
| Research 3 | 143 passed | independent clone `8893eec` |
| Research 4 | 213 passed, 1 skipped | independent clone `8893eec` |
| Sandbox top-level 1 / 2 / 3 | 227 / 129 / 159 passed | independent clone `8893eec` |
| Sandbox perception + track | 53 passed | independent clone `8893eec` |
| Runtime | 29 passed | independent clone `8893eec` |
| Track assets | 84 passed, 1 skipped | independent clone `8893eec` |
| Integration | 313 passed | independent clone `8893eec` |
| Learning | 1,188 passed, 1 skipped | independent clone `8893eec` |
| Bootstrap replacement tests | 3 passed | `616337d` |

The grouped Windows totals are 2,983 passed, 3 skipped at the earlier test version; replacing its two bootstrap tests with the three tested on `616337d` yields 2,984 distinct passed and 3 skipped. The only code change between those commits is the bootstrap and its tests. A single full-suite pytest summary on `616337d` was not produced; the earlier 600-second research-2 group timed out and was successfully rerun in the four subgroups above. No tests were omitted by that partition. Engineering regressions exercise fail-closed scientific gates; passing them does not authorize research evaluation.

The three Windows skips were symlink creation restrictions: track manifest, file fingerprint and WP2 nonce ledger. Those exact three tests were rerun on Ubuntu WSL, Python 3.14, using real Linux CPU dependencies and `PYTHONPATH` pointing to the checkout and temporary dependencies: **2 passed** for the track/fingerprint pair and **1 passed** for the nonce ledger; no skips. For example, run `python -m pytest -q -rs tests/track_assets/test_manifest.py::test_symlink_escape_is_rejected_when_supported tests/learning/provenance/test_file_fingerprints.py::test_symlink_escape_is_rejected_when_supported tests/research/test_wp2_execution_permit.py::test_nonce_ledger_rejects_symlinked_ancestor_before_claim` on a host that supports symlinks and has the project RL and science dependencies.

For a partition rerun on Windows, select each directory's sorted `test_*.py` paths by the slices above and invoke `python -m pytest -q <selected paths>`; run the directories `tests/integration`, `tests/learning`, `tests/runtime`, `tests/track_assets`, `tests/sandbox/perception` and `tests/sandbox/track` separately. The slow research-2 files are `test_wp7_coupled_ramp_execution.py`, `test_wp7_rotating_owner_execution.py`, and `test_wp7_spatial_profile_execution.py`; run them individually with the same command.
