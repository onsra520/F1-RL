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

# Remediation follow-up (historical, before final acceptance)

Engineering verdict at remediation checkpoint: **CONDITIONAL**. All grouped tests have final summaries and the restore path is demonstrated, but the groups were run across two successive commits rather than on one exact source tree and the broad semantic history review remains limited. Do not propose fast-forward on this evidence alone. Scientific registered evaluation and independent physics approval remain BLOCKED; no scientific permission is granted.

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

## Final Engineering Acceptance — 2026-10-28

Frozen source: `28ca37b2efe7caf765a0adde1a6413e531bb4b96`. `git fetch origin --prune` succeeded; branch and remote matched and the consolidation and architecture worktrees were clean before testing. The `main` worktree had an unrelated untracked `REAME.md`, left untouched. No source was edited during acceptance. This report is committed separately after the tests; its only change relative to the frozen commit is this audit file. No merge, rebase, training campaign, HELD-OUT access, GitHub Actions run or scientific authorization occurred.

### Model, evidence, history and runtime

- `python scripts/verify_consolidated_wp3_assets.py`: exit 0; 12/12 model SHA-256 and ZIP CRC, 7/7 reports, 96/96 cell hashes, canonical run-manifest and receipt-bound output-manifest hashes. Independently deserialized all 12 real models with `stable_baselines3.PPO.load(..., device="cpu")`, checked each model hash and `num_timesteps == 65536` (exit 0).
- Independent JSONL count: 96 cells, 5,760 started and 5,760 completed; finish 1,550, physical crash 1,253, hard-route violation 1,256, step limit 1,700, roll-envelope/tilt exit 1. These are recorded TRAIN-development outcomes, not new evaluation or scientific certification.
- Rechecked all 153 branch asset-inventory entries, and the handoff archive SHA-256 `585eabeba9003dfc5db389e53051c6a9d0c71d0a7b171a5ca2bdbaf95ec8a034`, ZIP CRC and 501/501 member hashes. Original three handoff manifests match 73/73, 216/216 and 204/204 entries. `git bundle verify` exit 0; quaternion source remains recoverable separately.
- Historical restore: `python scripts/restore_wp3_historical_evidence.py --full <WP3-attempt-v8-full-evidence.zip> --continuation <WP3-attempt-v8-continuation-9-telemetry-20261022.zip> --destination <new-directory>` exit 0 using the two immutable release archives whose SHA-256 are `fd39193a8689db1ba529910851435a46402981372b1dd893c7fd523b72363edf` and `726573b374f3aec727457f2c7b7b1f6533ae09079e257577b85ced17fac18926`. Restored directory: `D:/Projects/.evidence/F1-RL/final-acceptance-28ca37b-20261028`, no junction to old `.temp`; historical preflight verified 12 models/telemetry, roster `f2f9d15954cea48345a8932773ed945f6ed250ad1435b4e675b97e4a6b94f319`, and scientific gate remained BLOCKED. The bootstrap negative-integrity and existing-destination tests are included in research-1 and passed. Raw telemetry remains external; a full Git clone and these immutable ZIPs are needed to restore it.
- An independent no-hardlinks clone at `D:/Projects/.temp/F1-RL-final-acceptance-28ca37b` imported `f1rl` and Gym from that clone, exit 0. `python scripts/smoke_consolidated_wp3_model.py` on the frozen worktree loaded the real A-seed0 model and completed one TRAIN-development Physics/Gym/PPO episode: 500 steps, `step_limit`, zero checkpoints; exit 0. The integration and research suites cover actual checkpoint crossing, hard-route, crash and roll-envelope terminal contracts. No claim that this policy finished a lap.
- UI ↔ Python runtime bridge: **NOT IMPLEMENTED**, intentionally outside consolidation scope.

### Frontend and dependency exposure

In `web`, `npm ci`, `npm test`, `npm run typecheck`, and `npm run build` all exited 0. Vitest: 12 passed across four files; production build succeeded with a >500 kB chunk warning. Installation blocked optional install scripts for esbuild, but build succeeded. `npm audit --json` exited 1 and reported 6 affected packages: 3 moderate (`@vitest/mocker`, `esbuild`, `vite-node`), 1 high (`vite` nested under test tooling), 2 critical (`vitest`, `tinypool`). `npm audit --omit=dev --json` exited 0, 0 reported production dependency vulnerabilities. This is not a general security certification.

The lockfile resolves the reported findings under dev/test tooling. The Vitest advisory [GHSA-5xrq-8626-4rwp](https://github.com/advisories/GHSA-5xrq-8626-4rwp) requires the Vitest UI/API or browser mode to be listening; this project's `npm test` uses `vitest run` without UI and does not explicitly expose it. [GHSA-5gmw-xhrv-c9v3](https://github.com/advisories/GHSA-5gmw-xhrv-c9v3) describes code execution in test workers if an attacker first gains prototype pollution in the process; actual exploitability in this repository was not proven. [GHSA-67mh-4wv8-2f99](https://github.com/advisories/GHSA-67mh-4wv8-2f99) concerns malicious websites reading a vulnerable esbuild development server, including localhost; applicability of the specific nested version and launch path was not independently demonstrated. High Vite advisories concern development-server file access and Windows editor handling. Do not expose test UI or serve untrusted test inputs; plan a reviewed Vitest/Vite dependency upgrade. No breaking dependency upgrade was made.

### Branch and semantic decisions

`git fetch origin --prune`, ancestry tests and `git cherry` were rerun. No important source fix remains unclassified within the listed branches:

| Area / source | Decision | Evidence and contract |
| --- | --- | --- |
| `frontend/racing-sandbox` | INCLUDED | Ancestor; frontend suite/build run on frozen source. |
| WP1E `1af15fd`, `592be32`, `d8d0a03`, `5f1f217`, `4d4348a`, `97f7ffa`, `3510c20`, `af79543`, `a166742`, `b5e3969` | INCLUDED / EQUIVALENT | Ported behavior in `2598985` and `ae75f0e`; signatures alone do not authorize draft training; cleanup preserves primary failures; vector telemetry bound and exact rollout-multiple rule are retained. Real reset provenance is tested. |
| WP1E `b79406d` | EQUIVALENT | `git cherry` patch-equivalent. |
| WP3 tuning, evaluation, impact, roll domain and frontend integration | INCLUDED | Relevant source refs are ancestors, including active `fc18787` roll-envelope terminal and `6c47954` road impact. Engineering tests run on frozen source. |
| WP7 research and validation source refs | INCLUDED / EQUIVALENT | Local WP7 refs are ancestors; `ac330b0` patch-equivalent, `be63b0b` regression included via consolidation. Tests remain fail-closed for scientific admissions. |
| Handoff `97ac293` through `2ae51b6`; evidence `255bf39` through `879c124` | ARCHIVED / INCLUDED | Historical archives and inventories retained; source commits on evidence-only branch do not change application source; required 12 models, 26 small preflight inputs and canonical FULL-06 included. Other evidence remains restorable in verified archives. |
| Quaternion alternative `13b1d81` | ARCHIVED / REJECTED from active runtime | Its crossing behavior conflicts with the chosen absorbing roll-envelope contract; preserved in verified bundle, not declared equivalent. |
| Historical Archify `ef95ec5` | INCLUDED | Restored with historical-snapshot warning. |
| Scientific approval / HELD-OUT / registered evaluation | BLOCKED | Outside engineering acceptance; no inferred owner or scientific approval. |

### Final Python test matrix

Windows Python 3.12, pytest 9.1.1, numpy 2.5.4, stable-baselines3 2.9.0, cryptography 50.0.2. `PYTHONPATH` was the frozen worktree, `F1RL_WP3_EVIDENCE` the independently restored fixture above, and `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` were 2. No OS affinity limit. One WP7-heavy group ran at a time; a light suite ran concurrently. Each group invoked `python -m pytest -q -rs <its assigned paths>` with a 1,200-second per-group limit. The full driver, exact argument arrays, group-to-file inventories, exit codes and logs are stored under `D:/Projects/.archives/F1-RL/pre-cleanup-consolidation-818eedc/final-acceptance-28ca37b/`; the runnable driver is `D:/Projects/.archives/F1-RL/pre-cleanup-consolidation-818eedc/final-acceptance-driver.py`, invoked as `python <driver> heavy` and `python <driver> light` from the clean frozen checkout. It refuses to run if SHA or tracked/untracked status changes. It asserts 254/254 files, uniqueness and completeness before testing.

| Group | Files | Exit | pytest summary |
| --- | ---: | ---: | --- |
| integration | 38 | 0 | 313 passed in 49.53s |
| learning | 70 | 0 | 1188 passed, 1 skipped in 98.08s (0:01:38) |
| research-1 | 21 | 0 | 245 passed in 57.21s |
| research-2-fast | 18 | 0 | 177 passed in 27.08s |
| research-3 | 20 | 0 | 143 passed in 167.27s (0:02:47) |
| research-4 | 20 | 0 | 213 passed, 1 skipped in 548.41s (0:09:08) |
| runtime | 3 | 0 | 29 passed in 0.11s |
| sandbox-1 | 16 | 0 | 227 passed in 4.20s |
| sandbox-2 | 15 | 0 | 129 passed in 9.29s |
| sandbox-3 | 15 | 0 | 159 passed in 10.27s |
| sandbox-perception | 2 | 0 | 11 passed in 0.06s |
| sandbox-track | 6 | 0 | 42 passed in 0.08s |
| test_wp7_coupled_ramp_execution | 1 | 0 | 1 passed in 74.90s (0:01:14) |
| test_wp7_rotating_owner_execution | 1 | 0 | 4 passed in 260.42s (0:04:20) |
| test_wp7_spatial_profile_execution | 1 | 0 | 19 passed in 497.13s (0:08:17) |
| track_assets | 7 | 0 | 84 passed, 1 skipped in 1.33s |

**Windows aggregate: 2,984 passed, 0 failed, 3 skipped, 0 errors; 254/254 test files, 16/16 groups exit 0 and final summaries.** The three skips are symlink restrictions on Windows (track manifest, file fingerprints, WP2 nonce ledger). Those exact three were rerun on Ubuntu WSL/Python 3.14 using the same frozen worktree and real Linux CPU dependencies: `python3 -m pytest -q -rs tests/track_assets/test_manifest.py::test_symlink_escape_is_rejected_when_supported tests/learning/provenance/test_file_fingerprints.py::test_symlink_escape_is_rejected_when_supported tests/research/test_wp2_execution_permit.py::test_nonce_ledger_rejects_symlinked_ancestor_before_claim` — exit 0, **3 passed**. These are reruns of skipped tests and are not added to the Windows aggregate.

### Verdict and handoff

**ENGINEERING PASS** for frozen source `28ca37b2efe7caf765a0adde1a6413e531bb4b96`: all specified engineering gates passed, with Windows symlink limitations covered by WSL and external historical inputs restored and verified. The dev/test dependency vulnerabilities and external release availability are known limitations; the UI/backend bridge is outside this stage. There are no remaining engineering blockers identified in this acceptance. Scientific signoff, HELD-OUT and registered evaluation remain separately BLOCKED. The architecture branch `69acffa` is an ancestor of the frozen consolidation branch; fast-forward may be proposed but was not performed.
