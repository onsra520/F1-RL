# Pre-cleanup consolidation

Baseline: `818eedcdd03710735d763f8eb8d574599f2a605e` (`main` and `refactor/architecture-modularization`). Implementation tested at `d8be18f`.

## Decisions and source commits

- `1af15fd`, `592be32`: draft A/B/C/D is rejected at candidate and repository claim boundaries. A signature-only verification and low-level claim carry `training_permitted: false`; held-out remains false. The checked-in manifest validator only accepts DRAFT, so an approved protocol cannot currently be activated through this path. A future approved protocol must be independently designed and reviewed; in particular, the repository claim must independently bind the approved manifest hash and current source to the signed expected identity before the approved path is enabled. No scientific permission is granted here.
- `d8d0a03`, `5f1f217`, `4d4348a`, `97f7ffa`: close telemetry and environments independently, report finalization failures, preserve the primary error and attach cleanup errors as notes. PPO hook removal and telemetry disable are both attempted. A failure writing worker state is also attached to the original error.
- `3510c20`: vector capacity preflight is retained via the more conservative max-budget plus vector-batch bound in `97f7ffa`. Because SB3 completes whole `n_steps * n_envs` rollouts, telemetry-backed PPO calls must now request an exact rollout multiple before the observer or learning starts. This is an intentional fail-closed compatibility restriction.
- `b5e3969`, `a166742`: baseline already exercises declared versus actual real TRAIN reset transitions. `af79543` adds malformed/missing real-transition identity coverage. Admission verification also rejects a seed outside both rosters even when the claimed role is null.
- `b79406d` is already patch-equivalent in baseline and was not duplicated. `be63b0b` adds the missing non-contact obstacle sweep regression; production fix `ac330b0` is already patch-equivalent.
- `13b1d81` is not ported: it permits source-certified full-quaternion motion across the roll boundary, whereas `fc18787` establishes a certified, noncollision absorbing roll-envelope terminal. These are different behavioral contracts; no equivalence is established. Keep the old ref or verified bundle for later physics review.
- `ef95ec5`: historical Archify HTML and specification are restored with prominent warnings, explicitly pinned to `c0e1101`. React is present now but its Python bridge is NOT IMPLEMENTED. The snapshot is not a current architecture certification.
- Tests requiring frozen WP3 evidence accept `F1RL_WP3_EVIDENCE` as a test fixture path; the production scientific acceptance logic is unchanged. The original absolute default remains available; production gate defaults are still hardcoded and production-path portability is not established.

## Recovery and integrity

Archives are in `D:/Projects/.archives/F1-RL/pre-cleanup-consolidation-818eedc`; source data was not overwritten.

- `unique-branches.bundle`: 7 refs including both local WP1E branches, handoff, FULL-06 evidence, roll alternative, suspension and architecture. `git bundle verify` reports complete history. SHA-256 `58387b8755ed496ecb48630ad9abda15915794901a8aac624831678060b0a22b`.
- `evidence-wp3-87face7-raw.zip`: SHA-256 `144a41a5e2b2f393103321487265a6ba0f98ec1075196e4f796aecf72b3ef2ba`; 7/7 FULL-06 output files and 96/96 cell logs match the output manifest. CRC verified.
- `handoff-wp3-2ae51b6-manifest-verified.zip`: SHA-256 `585eabeba9003dfc5db389e53051c6a9d0c71d0a7b171a5ca2bdbaf95ec8a034`; handoff manifests match 73/73, 216/216 and 204/204 files. One stdout log was reconstructed using the CRLF bytes in the separately retained Windows archive because the raw Git blob was LF and differed from its manifest; the reconstructed bytes match the original recorded SHA-256. Both unmodified Git archive variants are retained.
- GitHub release `WP3-attempt-v8-full-evidence.zip`: SHA-256 `fd39193a8689db1ba529910851435a46402981372b1dd893c7fd523b72363edf`; continuation archive SHA-256 `726573b374f3aec727457f2c7b7b1f6533ae09079e257577b85ced17fac18926`. Both match release metadata; extracted contents have not been independently audited.
- Local ignored `ppo-straight-improved-16384.zip` SHA-256 `b3524af0eaeff3458293e2f59a271b590cb9a57a0c0f626569bd9cd1e6d3505f` and `ppo-straight-trained.zip` SHA-256 `b685284e35b39d094344aa9797f323c4def4666c808176ad89845d79994e660f` were copied into `local-models` and match source hashes.
- The test fixture at `D:/Projects/.evidence/F1-RL/WP3-EVIDENCE/wp3-eval-issue51-v1` is a junction to a temporary location; do not treat it as an independent durable backup. No Git objects or evidence refs should be deleted solely on the basis of an archive integrity check.

## Gates

- Safety: 105 passed, 1 skipped, on `ae75f0e`.
- Lifecycle: 119 passed on `ae75f0e`; includes observer and PPO wrapper behavior.
- Physics v4: 158 passed on `ae75f0e`; covers suspension, collision, airborne, slope, roll, crash terminal and issue-51 gates. An additional 21 checkpoint/interactions, world, crash-prefix and slope tests passed on `d8be18f`. This is a selected gate, not exhaustive physics validation.
- RL integration: 255 passed on `d8be18f`; PPO, Gym, rewards, reset/step and report tests; another 23 real PPO/vector integration tests and 49 WP3 coordinator/admission tests passed on `d8be18f`.
- Frontend: 12/12 unit tests, TypeScript typecheck and production build passed. Build warned about a chunk exceeding 500 kB. UI–Python bridge: NOT IMPLEMENTED.
- Full Python + WP7: INCOMPLETE after 300 seconds on `d8be18f`; last progress marker 69%, with 2,122 success markers and 2 skip markers, no failure markers, and no final pytest summary. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` set to 2; 300-second wall-clock limit; pytest output in `full-python-wp7-d8be18f.log` outside source. Thread settings are not an OS CPU-affinity limit; a separate monitor stops the test process if the log exceeds 1 MiB (checked once per second). First attempt on `3f8081f` reached 70%, 23 failures due to paths requiring WP3 evidence; it timed out before final summary. Those paths were fixed and corresponding focused tests passed. Second attempt was interrupted at 4% to fix the PPO backend test double. No complete-suite PASS is claimed.
- Runtime smoke: loaded original A-seed0 `model-final.zip` (SHA-256 `8cfcde6d4c88bbcc01dbf9b67dee3829c991474b0a8883a4479370f665c01c79`, 65,536 prior steps), checked against its state manifest, ran five simulator/Gym PPO inference steps and one TRAIN-development evaluation episode. Episode hit 500-step limit, 0 checkpoints, no finish. No new PPO training campaign, HELD-OUT access or scientific approval occurred.

## Cleanup after acceptance

Do not remove `main`, the architecture-modularization or consolidation worktrees, or PR #52–#55 head/base branches while those PRs remain open. Preserve the WP3 evidence and handoff branches until independent restoration from the bundle and release archives has been verified and research owners approve. Keep the roll alternative until its different physics semantics are reviewed. WP1E source branches can be considered only after the consolidation branch is accepted and the bundle is independently restored. Ancestor branches identified as safe in the pre-cleanup audit may be reviewed separately, after rechecking PRs, worktree state and any named external links. This consolidation does not authorize cleanup.