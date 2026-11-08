# TRAIN-5 v4 readiness and protocol review

The proposed protocol is `configs/research/wp7-train5-v4-simulation-protocol-draft.json`. It records simulation-only scope, 18 newly proposed training seed assignments, the ordered C-seed0/B-seed1/A-seed2 frozen initializers, the final-only 200,704-step budget, fixed development thresholds and the planned policy-clustered analysis. The 15 unavailable historical job logs remain unavailable; the exception for the old 18 seeds does not authorize or prove non-use of the new seeds. The new seed registry remains open. The draft is not a signed owner decision or an independent scientific review. Statistical power and real-world calibration are not established.

The draft binds all Python files under `f1rl/` and the registered source, TRAIN asset and model roster by SHA-256 (canonical LF for text, exact bytes for ZIP archives). This is a working-tree integrity check, not an independently authenticated source or installed-dependency attestation. Development seed derivation uses the saved protocol hash and checks all 3,600 proposed values against each other and the training values; it does not establish the historical non-use of those development seeds, run an episode or consume a training seed. The evaluator checks local run state, seed/track labels, recorded development seeds, model hashes and frozen initializer assignment. Local state and records alone are not an independently authenticated run receipt.

The original consecutive PPO seeds create overlapping `ppo_seed + environment rank` ranges with eight environments. At the owner's direction a separate proposal fixes 18 deterministic new values with non-overlapping vector ranges, distinct from the old values and their ranges; the historical protocol remains unchanged. The bounded local scan searched 60 new base/rank values in 34,236 readable text files in 80 F1-RL directories (562,697,139 bytes), with zero matching files or read errors; 24 files larger than 10 MiB, archives, Git objects and external/deleted histories were excluded. A separate `git log --all -G` search of reachable local clone history found zero matching commits. Google Drive full-text index searches for all 60 values returned no files; deleted and unindexed content and past execution state are outside that check. These results do not establish global non-use. The audit SHA-256 is recorded in the proposal; seed registry review, the inaccessible logs and owner/independent attestation remain open. Development seed collision checks include both proposed and historical vector-rank values.

| Subsystem | Compatibility gate |
| --- | --- |
| simulation_runtime | IN_PROGRESS |
| vehicle_dynamics | IN_PROGRESS |
| track_feasibility | IN_PROGRESS |
| race_episode | IN_PROGRESS |
| gym_reset | IN_PROGRESS |
| perception_control | IN_PROGRESS |
| train_ppo | IN_PROGRESS |
| evaluation | IN_PROGRESS |
| provenance_admission | IN_PROGRESS |

All nine remain open pending subsystem-specific exact-source Windows/Linux evidence, historical C2 trajectory replay and cross-module seed/schema evidence. The repository has a historical C2 draft configuration and contract tests, but the bounded filename and text search of the available handoff, WP3 and Physics-v4 evidence directories found no verified original C2 trajectory/action tape and replay receipt. Archived ZIP contents and other unnamed locations were not exhaustively searched. A configuration restore or a synthetic replay is not the missing historical trajectory proof.

The v4 admission assessor and registered v4 runner always deny training and HELD-OUT, including when signature bytes match a repository registry. The registry is not independently authenticated here, and the supplied run-manifest hash is not a verified manifest. The historical v2, runner and PPO guards continue to deny research training. The registered runner checks track names and reset labels but does not yet bind the actual factories to the registered geometry and full Physics/action/observation/reward contracts. The existing v3 fixture reset adapter deliberately rejects the 18 reserved training seeds, so it cannot be used as the training scenario builder. A future admission implementation must resolve these integration gaps and bind an actual run manifest, independently authenticated reviewer keys, scientific protocol and source hashes across all training guards. None of those requirements can be replaced by setting approval booleans.

The CI workflow is configured for the handoff and v4 work branch on Windows and Linux. Its results apply only to the exact committed source that ran; an uncommitted local test or earlier workflow result cannot close this gate. CI and software tests do not establish physical calibration, statistical power or independent scientific approval.
