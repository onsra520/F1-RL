# TRAIN-5 v4 factory preview and evidence follow-up

## Scope and decision

This is a source-only engineering preview, not a training permit or scientific acceptance. `scripts/wp7_train5_v4_factory_preview.py` validates the committed v4 protocol and the cross-platform 44-source/four-witness preflight, reconstructs the registered five TRAIN factories, and attaches the proposed perturbation seed without calling reset or step. It constructs and closes Gym environments when `TrainingScenarioSet` checks space compatibility; it does not load a model or execute an episode. Factory comparison rebuilds an independent reference and rejects differences in geometry, Physics, action, observation, reward, reset and other factory fields. Both the separate factory-admission boundary and the registered runner remain fail-closed.

The preview script is committed separately but is not included in the 221-file lock of the v4 protocol. The separate unsigned review manifest binds the adapter and tests for engineering review. This comparison is engineering evidence, not an authenticated source-bound training permit. Before admission, a new reviewed protocol must lock the adapter and tests, bind real scenario objects and runtime/environment identities to the signed run manifest, and connect a separately authenticated permit through every training guard. Existing v2, runner and PPO guards remain unchanged.

On the local machine, the five factory preview and comparison completed with all four external historical Step-3 witness hashes verified. A separate valid-factory check and the registered-runner boundary both denied training before model creation. No reset, physics step, PPO training, HELD-OUT geometry, or evaluation was executed. Portable tests mock the external witness receipt; they are not a substitute for the local four-witness verification.

The updated engineering-preflight test selection passed locally on Windows: 469 tests at that time. The draft protocol file SHA-256 is `2df3b500df5e50f6153655492f064d9823c6334f2462079bb5d6b5a21ea2531d`; the preview and its test had SHA-256 `57867fe5554b3837030abee7edbf206b8b07f0b18d05b262ffd037f97ad3ab3e` and `917e837bab87f9cf518a97766e8073d8b32e82898bdfed3343fc752be2f2a5e1` respectively at test time. These hashes identify the material for review, not an approval.

## Seed-history follow-up

The new 18 base values and their 42 additional PPO vector-rank values remain a proposal. The previous scan covered 34,236 text files, local reachable Git history and the accessible Google Drive full-text index. The follow-up checked the 24 omitted large on-disk telemetry files: 6,477,690,902 bytes; no matches were reported. Twenty-four large telemetry entries in three archives were SHA-256 identical to those scanned files. In 695 readable evidence/backup ZIPs, 3,009 text entries produced no matches; 185 `.zip`-named files were not readable as ZIP. Of 288 nested archives, 286 were inspected, while the remaining two were byte-identical to separately checked archives. The earlier receipts report 60 searched values but do not enumerate them, so they cannot independently establish that all 42 were derived from PPO seeds; that requires a reproducible recheck. This is bounded evidence, not a full binary or global history search.

Correction: the first follow-up scan of the 12 deeper nested archives incorrectly derived 42 ranks from perturbation seeds. It has been repeated using PPO seeds plus ranks 1–7 and now records the complete 60-value list. The 12 archives contained 12 text members (2,232 bytes); there were no exact candidate seed matches or C2/reviewer filename candidates in this bounded scan. The earlier erroneous receipt is retained for audit and must not be used as evidence for PPO vector-rank coverage.

GitHub Actions job-log follow-up reported checking 60 values in seven accessible historical jobs with no match; the receipt does not record the exact values. Fifteen selected historical job logs still returned GitHub 404 `BlobNotFound`; other unavailable, deleted, non-indexed, offline or cloud histories remain outside scope. The old seed exception applies only to the old 18 values. The new seed registry remains open, no global non-use is claimed, and no proposed training seed has been consumed by this preview.

Further metadata checks on the 15 jobs found that all report failure, step details are absent, and their 15 workflow runs each return zero retained artifacts. Metadata cannot substitute for the lost logs. A corrected local 60-value search found 54 pytest fixture state files and six draft/audit references; 12 large local telemetry files produced no matches. The fixture files are synthetic state created by `test_metrics_bind_to_the_six_saved_final_checkpoints`, not evidence of completed training. These results do not establish global seed non-use.

Read-only local receipts, outside the repository:

- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-LARGE-TELEMETRY-SEED-AUDIT.json`, SHA-256 `438ae719ef8532b060ec4520126d63b9ffedf71dbb59ae940d42ac0970ca0f55`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-ARCHIVE-SEED-AUDIT.json`, SHA-256 `17d23fde41bfb33df4766ced3fe790e243e1e4c48b54a2fee3566510b850718a`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-NESTED-ARCHIVE-AUDIT.json`, SHA-256 `f8bf616ee307a45391ff608e12f92813a368f826b598755ab6a51f51623ed475`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-DEEP-ARCHIVE-PPO-RANK-AUDIT.json`, SHA-256 `53d8b51336013d5abc507dad4979f0a686b9e6791c97fc25d8883287fb1eeeb7`; generated by `D:\handoffphysics-v4-khanh-bang-20261106\scan_deep_archives.py`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-LARGE-ARCHIVE-TELEMETRY-AUDIT.json`, SHA-256 `b1aaec292ac1b38d85bcf635b82dfdd5fbd852636789542ce10e4aeb36d3f9d2`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-ACTIONS-SEED-AUDIT.json`, SHA-256 `b4a0ac7a00484ae27e244b0485e5e040894c11621da08142db79f5eec76584bf`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-UNAVAILABLE-JOB-METADATA-AUDIT.json`, SHA-256 `7646d2f10aa38776f3166d80b84919082c52040930a35e91069330f56eda8408`.
- `D:\handoffphysics-v4-khanh-bang-20261106\F1RL-V4-CORRECTED-LOCAL-SEED-AUDIT.json`, SHA-256 `1f9ac36db4f51d6a9584dc31284350315ce323cab2cf637ddb60b53ac75b1464`; classification receipt `F1RL-V4-CORRECTED-SEED-HIT-CLASSIFICATION.json`, SHA-256 `2d1559130bdecc8b0fa64db24552fbc5114ca91aa28d8e3b68f602663c6ecb4b`.

## C2, reviewer identities and nine gates

The historical C2 trajectory report points to `D:\Projects\.archives\F1-RL\wp7-exploratory`, which does not exist on this machine. Filename searches in reachable Git history, the accessible F1-RL directories, 695 outer ZIPs, 286 inspected nested ZIPs and 12 deeper nested ZIPs found no matching original C2 action tape or independently trusted replay receipt. Historical reports and a C2 draft configuration do exist; they cannot establish a reproducible exact historical replay by themselves. Other locations and deleted histories were not exhaustively searched.

The GitHub default-branch `configs/research/wp7-trusted-reviewers-v1.json` has an empty `reviewers` list. All 32 matching local registry files and two matching nested-archive registries inspected were also empty. No genuine independently authenticated reviewer key, owner/reviewer signatures, or training-specific permit was established. A review signature is not itself an execution permit.

The user identified themself as the prospective reviewer and confirmed direct participation in designing or implementing the TRAIN-5 protocol/runner. The user subsequently confirmed no independent reviewer is currently available. An independent reviewer separate from that work remains necessary; the user's statement is not an independent review signature or admission. A local SSH Ed25519 public key labelled `uav-t3-owner` was found at `C:\Users\KHANH BANG\.ssh\uav_t3_owner.pub`, with OpenSSH SHA-256 fingerprint `SHA256:6S0rHCQyWkbb/aV+GyxpAm4nSjDPbtdmD7Eh9CYZrVY`. Its ownership and authorization for this research have not been independently established; its name alone does not establish either, and it has not been added to the trust registry or used for signing.

GitHub Actions run `42503157205` passed 440 tests on Windows and 440 on Ubuntu for commit `484c306dd0a9d82907cfba8e11c31674be66dfe5`; it did not include the preview. Run `42507231870` subsequently passed Windows and Ubuntu on commit `f2cc663ccd7c54be697b68d530eee0ec4db616c9`, including the preview. Neither run establishes complete nine-gate acceptance; check CI separately for later candidate commits. The following gates all remain `IN_PROGRESS`:

| Gate | Evidence and remaining requirement |
| --- | --- |
| simulation_runtime | Exact-commit CI subset; historical C2 and per-gate cross-module evidence open. |
| vehicle_dynamics | Exact-commit CI subset; historical C2 and per-gate evidence open. |
| track_feasibility | Exact-commit CI subset; factory preview is engineering-only; historical C2 open. |
| race_episode | Exact-commit CI subset; historical C2 and per-gate evidence open. |
| gym_reset | Exact-commit CI subset; new adapter is preparation only; validated actual reset evidence open. |
| perception_control | Exact-commit CI subset; factory comparison is engineering-only; per-gate evidence open. |
| train_ppo | Exact-commit source-only tests; source-bound execution permit and admitted runner integration absent. |
| evaluation | Exact-commit software tests; authenticated run/evaluation evidence and stochastic admission absent. |
| provenance_admission | Exact-commit software tests; trusted reviewer bootstrap, authentic run manifest and permit absent. |

No six-run TRAIN-5 training or scientific/HELD-OUT release is authorized or performed.
