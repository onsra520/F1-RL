# F1-RL Consolidation Acceptance Report

**Date:** 2026-11-09

**Repository:** `D:\Projects\F1-RL`
**Verdict:** **MAIN_NOT_READY**

## Acceptance summary

The local `main` now contains the canonical `f1rl/` source and the consolidated runtime, frontend, Track Builder, and contract/test surface. The legacy `src/` tree and `uv.lock` are absent from the committed source. The clean Python 3.12 environment, package build, frontend checks, backend startup, and focused engineering checks passed.

The guarded full Python regression completed with **3,135 passed, 100 failed, and 91 skipped** (3,326 tests; 35:21). Failures include absent external WP3 evidence, stale source-hash registrations, two cases blocked by the required audit guard, and seven runtime/test-contract failures. The remaining failures prevent acceptance as ready.

No training, retraining, HELD-OUT evaluation, or FULL-06 scientific evaluation was run. The audit harness blocks training and production scientific evaluation. The Băng worktree was not modified.

## Source provenance and parity

| Source | Ref and inspected HEAD | Consolidation result |
|---|---|---|
| Băng Physics v4 handoff | Local worktree `handoff/physics-v4-khanh-bang-20261106`: `1de279c37e62a27531d661480d20fa6e095c4e84` (left untouched). Latest fetched authoritative remote `origin/handoff/physics-v4-khanh-bang-20261106`: `c70d1c9c4a222c5378debe90facac3f634b2e377`. | Kept Physics v4, collision, track/vehicle dynamics, training contracts, and research guards. Restored five registered source files byte-for-byte from the remote handoff. Two other handoff source-lock entries are stale even against that handoff; details below. |
| Current canonical integration | `integration/f1rl-v4-consolidated`: `99c5313c8bf98c164518b880e626383ad7f1c6f5` | Ported canonical `f1rl/`, runtime API/session, Track Builder, PPO integration, Auto Drive/Hybrid, replay and logging, frontend, tests, configs, and runtime consumers. Integration commits: `6cdead0fc7fcba1cd61895f3872433a60e8509b2`, `99c5313c8bf98c164518b880e626383ad7f1c6f5`. |
| RC FULL-06 | `origin/rc/physics-v4-train5-v2-review-20261107-full06-eol`: `c6c2c731b6237ef8a8fcfa4517d298565afcff61` | Inspected for portable contracts and source not already present. Reusable source/contracts were retained where compatible with the Băng handoff; scientific evaluation and FULL-06 execution were excluded. The complete RC snapshot remains archived. |
| Older integration candidate | `integration/physics-v4-train5-20261108`: `14114181051daaf0f44221cffbfae4928203a1e6` | Compared against the canonical tree. Only missing supporting assets, scripts, and contracts with active consumers were restored; duplicate legacy runtime source was not retained. The branch snapshot remains archived. |
| Old `main` | `main` before replacement: `14114181051daaf0f44221cffbfae4928203a1e6` | Used for parity and backup. Retained current assets, models, configs, tests, and contracts with consumers; historical evidence is archived. Removed legacy `src/` from active source. |

### Băng source integrity

The following files on new `main` match the authoritative remote handoff byte-for-byte:

- `pyproject.toml`
- `f1rl/learning/pilot_preflight.py`
- `f1rl/learning/reconstruction.py`
- `f1rl/learning/training/env_factory.py`
- `f1rl/learning/gym_environment.py`

No Băng worktree or branch was edited. The source-bound WP7 gates still fail closed on two registrations whose expected hashes do not match Băng's latest committed source:

| File | Registration expects | Handoff Git blob at `c70d1c9` |
|---|---|---|
| `f1rl/research/wp7_track_feasibility.py` | `18ffc3050fe278cd62a9992ae2a62aef8876d577f4aabbfb10d9b9f4f5f55ab0` | `a0b6b53080f6e48d07993bbafb33da0e88537f40efb3afc0ea2ca8e10f5e64ab` |
| `f1rl/research/wp7_train5_v2_protocol.py` | `dda7d06759ef62cfb8b95b37c6b6bee3a49bbfab633bd2dfcbfb9f48f236c34e` | `2a3abe43eb5ed4c18c98b5263125dd5ae95bdd6405d37da4a7ddb12d07f45707` |

The Stage-2 proposal also binds `f1rl/learning/algorithms/ppo.py` to `d17b6edd30068673d5cf156cbf601a63a7e43e43726ad118981b05aafe84c4e6`; the current canonical file's LF-normalized SHA-256 is `68ff35f0c144bf2f8f4e75d1d06ff7f2da2ea2d35e9e65c71f2e22b0aa242d2e`. These bindings were not rewritten to make source-identity gates pass.

## Consolidated changes

- Replaced active legacy source with canonical `f1rl/`; removed the old `uv.lock`. The legacy `src/` files were absent from Git after consolidation; final cleanup also removed its empty leftover directories.
- Kept `web/`, runtime and scientific contract tests, current configs/assets/models, scripts, examples, and docs/evidence still used by active consumers.
- Archived historical docs/evidence/models and old source snapshots under `.archives` or the external archive root.
- Added `requirements-runtime.txt` and `requirements-test.txt` for runtime/API and test dependencies omitted from the project extras; updated installation guidance.
- Runtime adapter now exposes the state used by replay and Auto Drive; Track Builder surface patches meet at segment boundaries; failed runtime sessions can explicitly restart after their worker exits.
- Removed the stale `legacy_v3` runtime default in favor of the registered factory profile. Updated runtime assertions to preserve Băng's V4 collision behavior and fail-closed unsupported stacked-deck behavior.
- No Physics or PPO algorithm was changed to force regression tests to pass. Historical and scientific contracts were not rewritten to bless changed source hashes.

## Commits

- Consolidation branch: `6cdead0fc7fcba1cd61895f3872433a60e8509b2`, `99c5313c8bf98c164518b880e626383ad7f1c6f5`.
- New local `main`: `ac7bfc77094fdacc2ed7dd92f8625d84c02d0449` (`Consolidate canonical runtime and Physics v4 on main`).
- No push was performed.

## Archives and recovery

Archive root: `D:\Projects\.archives\F1-RL-final-consolidation-20261109`

Full manifest: `D:\Projects\.archives\F1-RL-final-consolidation-20261109\backup-manifest.json`

Every registered source ZIP in the manifest passed CRC and restore checks. ZIPs excluded `.venv`, `node_modules`, caches, and credentials.

| Snapshot | ZIP SHA-256 | CRC / restore |
|---|---|---|
| Old main `14114181` | `24359133b327c8866826b0b7a7a79f41f7620bd781dabc94cbb82837910a67e1` | PASS / PASS |
| Băng local worktree `1de279c3` | `8c2568bf1df98a373eba567a6e8b3fbc77d3cb3a66ddf84ba51b3ad570c82c40` | PASS / PASS |
| Latest remote Băng handoff `c70d1c9c` | `25857fd5cb91c97a80d76586c3c000f91e9b79356404821b0480a05b4641bf0d` | PASS / PASS |
| Consolidation worktree snapshot at `c70d1c9c` (including working changes) | `83d034da018dc0dad9eb34e00826c54179f3e7a1d06765aedf1693300edbed3e` | PASS / PASS |
| RC FULL-06 `c6c2c731` | `ff0ec87781eb17e8c5d4fc250a8ad2312da716406074753aa91c1d00a8b347c7` | PASS / PASS |
| Older integration snapshot `14114181` | `51091fd31393150eebb3f136e28c684d659433897291ade4814ebb2a34f79ac8` | PASS / PASS |
| Historical evidence/models (2,320 files) | `8abad359f3a5381e40837adee24fed6b71763b90de6c15721d237de4ce64ab1d` | PASS / PASS |

Complete Git history bundle: `D:\Projects\.archives\F1-RL-final-consolidation-20261109\f1rl-all-refs.bundle`

SHA-256: `38a4c6e088fc1a9a9510f3f76c7089a360848348b3375162f49a6a5899a82103`

Verification: **PASS**, complete history, 126 refs. The bundle includes the original main, integration, handoff, and RC refs; the manifest records their SHAs.

## Clean runtime and verification

Environment: Windows, CPython **3.12.15**, `uv`.

| Check | Result |
|---|---|
| `uv venv .venv --python 3.12` | PASS; fresh `.venv` |
| `uv pip install --python .venv\Scripts\python.exe --editable ".[dev,rl]" -r requirements-runtime.txt -r requirements-test.txt` | PASS; 46 packages installed |
| `uv pip check --python .venv\Scripts\python.exe` | PASS; no incompatible dependencies |
| `uv build --wheel --sdist` | PASS; wheel and sdist built |
| Imports of `f1rl`, PPO, runtime, and Track Builder | PASS |
| Uvicorn loopback `GET /api/v1/health` | PASS; `status=ok`, development runtime, scientific admission disabled, `f1rl_runtime_v1` |
| Frontend `npm ci` | PASS; 164 packages, zero reported vulnerabilities; deprecation and install-script permission warnings |
| Frontend `npm test` | PASS; 11 files, 105 tests |
| Frontend `npm run build` | PASS; Vite build completed; emitted bundle is about 921 KB and triggers the 500 KB advisory |
| Focused runtime/Track Builder/guidance/collision checks | PASS at checkpoints: 61 passed, 2 deselected; Track Builder 34 passed; collision terminal 2 passed; guidance 3 passed; Slalom V4 TRAIN 1 passed |
| Focused source-bound suite | PARTIAL: 41 passed, 7 failed in 48.01s; all seven fail closed on the stale `wp7_track_feasibility.py` source pin |

Backend startup was exercised against loopback and the server process was stopped afterward. The architecture JSON/HTML artifact was structurally validated and delivered; a browser renderer could not launch, so it was not visually reviewed.

### Guarded full Python regression

Command: `.venv\Scripts\python.exe .archives\tools\run_guarded_python_regression.py`

Result: **FAIL — 3,135 passed, 100 failed, 91 skipped in 2,121.57s (35:21)**.

JUnit: `.archives\audit\full-python-regression-guarded-final.xml`

JUnit SHA-256: `3283bcdff23802a9d6ef56231a681dc4d9636e6190a5790dfa3c4c1f9ea1b390`

Failure groups from the JUnit record:

| Count | Classification | Evidence |
|---:|---|---|
| 77 | External WP3 evidence unavailable | Tests require `D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1` and its evaluation roster, which are absent. No scientific evaluation was run to create them. |
| 14 | Source-lock/source-binding mismatch | 13 WP7 source-lock failures on the stale `wp7_track_feasibility.py` pin, plus one Stage-2 proposal binding mismatch for current PPO source. Exact hashes are above. |
| 2 | Blocked by the audit guard | WP3 smoke production-episode tests reach the harness guard, which raises `BLOCKED_EXPECTED` before evaluation. These are intentionally blocked under the no-evaluation constraint. |
| 7 | Runtime/test-contract failures | Two `F1RLEnvFactory` tests expect a `research_admission_required` constructor field absent from the canonical factory. Five Auto Drive sessions exceeded their 120s wait: cone/barrier obstacle runs and three T05 route/hitbox runs. |

The Auto Drive session tests fail at their wall-clock wait before replay assertions. A direct adapter/controller diagnostic for the 200 m cone track advanced 44 simulation steps in 20.42s and was still grounded at `z=25.08 m`, speed `8.07 m/s`, before the obstacle at approximately `z=120 m`; this indicates low local simulation throughput but does not establish whether the later timeout is caused only by throughput. The five timeouts remain unresolved and are included in the readiness blockers.

## Readiness blockers

1. WP7 source-lock registrations and the Stage-2 PPO binding do not match the source bytes under review. The gates correctly fail closed; changing the registered hashes would alter source-bound evidence and was not done.
2. External WP3 evidence required by 77 tests is not present locally. The acceptance run did not synthesize or execute scientific evidence.
3. Five Auto Drive/T05 runs time out under the test's 120s wait on this machine. The direct diagnostic was inconclusive about the eventual episode outcome.
4. Two environment-factory tests refer to a constructor option not implemented by the canonical factory. Whether that option is a required integration feature or stale test expectation needs a source/contract decision.

Because the full engineering regression is not green and these gates remain unresolved, the local consolidated `main` is **MAIN_NOT_READY**. It is committed locally as `ac7bfc77094fdacc2ed7dd92f8625d84c02d0449`; no remote push was made.
