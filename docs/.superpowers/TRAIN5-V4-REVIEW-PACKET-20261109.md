# TRAIN-5 v4 review packet — draft, admission closed

This packet prepares a future independent review. It is not an approval, signed run manifest, execution permit or authorization to train. The owner reports that no independent reviewer is currently available. The source tree below contains uncommitted additions; do not treat previous CI as testing them.

## Material to review

- Base commit: `484c306dd0a9d82907cfba8e11c31674be66dfe5`. GitHub Actions run `42503157205` passed 440 tests each on Windows and Ubuntu for this commit only.
- Draft protocol: `configs/research/wp7-train5-v4-simulation-protocol-draft.json`, file SHA-256 `2df3b500df5e50f6153655492f064d9823c6334f2462079bb5d6b5a21ea2531d`. Its own permissions remain false. Recompute all hashes when preparing the version for signature; this file hash is not an authenticated signature or necessarily the protocol's internal canonical digest.
- Unsigned review manifest: `configs/research/wp7-train5-v4-run-manifest-review-draft.json`, SHA-256 after CRLF-to-LF normalization `aeae574104eba98a53e254d05c0c2b2dd821be34c0d3aef969c5239351ef7875`. It binds the saved protocol, review adapter and tests, six run allocations and three raw checkpoint hashes. This is not an authenticated execution manifest or admission.
- Working-tree factory preview: `scripts/wp7_train5_v4_factory_preview.py`, SHA-256 `57867fe5554b3837030abee7edbf206b8b07f0b18d05b262ffd037f97ad3ab3e`; tests: `tests/research/test_wp7_train5_v4_factory_preview.py`, SHA-256 `917e837bab87f9cf518a97766e8073d8b32e82898bdfed3343fc752be2f2a5e1`. The adapter is not part of the committed protocol's source lock and is not a training runner.
- Evidence, search limits, all nine open gates and local receipt locations: `docs/.superpowers/TRAIN5-V4-FACTORY-EVIDENCE-20261109.md`. The updated Windows preflight selection passed 469 tests locally; it has not run in cross-platform CI on a commit containing these additions.
- Existing review verifier: `f1rl/research/wp7_trusted_reviews.py`. It verifies owner and independent reviewer signatures over a protocol and run-manifest digest against a separately authenticated registry. Its output explicitly does not permit training.

## Decisions required before execution

1. An independently authenticated owner-controlled registry must identify separate owner and independent reviewer keys. Verify identity, role, key validity and conflicts of interest out of band; a local key filename is not proof. The owner participated directly in TRAIN-5 work and cannot supply the independent review alone.
2. Review and approve the simulation-only scope, five training tracks, six-run design, initial checkpoint mapping, seeds, evaluation thresholds, statistical limitations and analysis in the protocol. In particular, determine whether the proposed 60 episodes per cell and three policies per cohort support the stated descriptive conclusions; do not infer statistical power from a software test.
3. Review the seed-history receipts and decide how to address the 15 inaccessible historical logs and other unavailable histories for the **new** seeds. Earlier receipts do not enumerate all searched values and cannot independently establish correct PPO-rank coverage; the deeper archive scan was corrected and rerun. The accepted exception for historical seeds does not extend to the new set. No global non-use is established.
4. Review all nine compatibility gates on the same candidate source. CI is partial evidence. The original C2 replay/action tape has not been recovered; any alternative or exception needs its own documented, legitimate decision, rather than a claim that replay passed.
5. Lock the adapter and its actual geometry, physics, action, observation, reward, reset, runtime and checkpoint bindings in the reviewed protocol and run manifest. The current preview compares reconstructed factories but is not a source-bound admission path. Verify checkpoint hashes and immutable originals at the actual loading boundary.
6. Prepare and validate a real run manifest for the exact candidate, then obtain the required authenticated owner and independent scientific reviews. Connect a separately authenticated, training-specific permit through all guards and verify fail-closed behavior for missing, modified, expired and wrong-scope evidence. Review signatures alone do not grant execution rights.
7. Commit the reviewed candidate and run Windows/Linux CI for that exact commit. Training remains blocked unless all applicable evidence and authorization requirements pass. HELD-OUT remains a separate decision.

Current state: **no independent reviewer, no authenticated training permit, no six-run training**.
