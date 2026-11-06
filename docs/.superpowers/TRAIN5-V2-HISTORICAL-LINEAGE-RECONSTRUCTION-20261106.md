# F1-RL TRAIN-5 v2 — scientific source lineage reconstruction

Status: **local review-only reconstruction; no training / PPO smoke / HELD-OUT**.

The isolated 2-commit Physics-v4 handoff snapshot at
`1de279c37e62a27531d661480d20fa6e095c4e84` was a *separate root history* and **did not historically
descend from WP3 commit `ce0e55d7a428826b2d4ed3218641105bd0206f4b`**. We do NOT claim otherwise.

To preserve the old WP3 trust prerequisite, this review worktree was actually
created from tracked Git commit `175e32d18ee0cc3a79a586d272d43857f1175d12`
in the original onsra520/F1-RL history. That commit has genuine Git ancestors:
WP3 handoff `ce0e55d7a428826b2d4ed3218641105bd0206f4b` and frozen TRAIN `fe83f517479f3b990028d6409304fae83b8199a9`.

The proposed *new commit* records the reconstructed source snapshot as one
explicit set of file changes on that true historical lineage. This is not
a cherry-pick of the original disconnected snapshot and not an in-place
rewrite of any old commit or original WP3 evidence.

- Source input: independent source worktree `D:\F1-RL-PHYSICS-V4-TRAIN5-RC-20261106` at `1de279c37e62a27531d661480d20fa6e095c4e84`,
  **582 copied selected source and frozen checkpoint files**, SHA-256 verified.
- The original 12 model-final ZIPs retain exact source bytes; they were
  copied, not trained, loaded or replaced in any original checkout.
- Historical HELD-OUT geometry is explicitly excluded by sparse-checkout.
  No original HELD-OUT asset file was materialized or opened for this reconstruction.
- The source-bound draft still protects 44 source SHA-256 and 4 externally
  locked Step3 receipt identities. No seed audit or scientific training approval
  is asserted merely from a valid Git parent.
- Old source tree files not in the disconnected snapshot may remain in this
  historical descendant's working tree; their provenance is the original
  historical HEAD `175e32d18ee0cc3a79a586d272d43857f1175d12`. The candidate tree is a NEW
  combined review state, not the historical old project's original tree.

Acceptance requires the original three WP3 ancestry tests and the two clean
WP2 draft-preparation tests to pass on a single newly committed, clean tree;
separate full regression, global seed attestation and GitHub CI remain required.

No push, merge, tag, production approval, PPO smoke or retraining is authorized
by this document or the act of committing the source reconstruction.
