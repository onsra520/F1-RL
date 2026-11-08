# WP7 compatibility, TRAIN-5 seed registry search and source/CI readiness

**Local audit status as of 2026-11-06:**

- Historical WP7 missing-file compatibility: **PASS**.
- TRAIN-5 v2 source-bound preview and local scoped regression: **PASS**.
- Exhaustive global seed history audit: **NOT COMPLETE / no owner signature**.
- GitHub Actions against an exact clean committed candidate: **NOT RUN** (all changes remain uncommitted).
- Six frozen PPO turn smoke episodes, retraining, HELD-OUT and FULL-06: **NOT RUN**.

## Historical WP7 restoration

Two files were absent from the minimal handoff checkout, causing two of 24 previous compatibility tests to fail:

1. `configs/research/wp7-spatial-ground-protocol-v1.json` — restored from **tracked Git blob**, historical source checkout `D:\F1-RL-WP3`, commit `fe83f517479f3b990028d6409304fae83b8199a9`. Canonical LF blob SHA-256 `7789babeb1317926fab6ad4ed1056438ba264c9936bb8bd990f6ee844f56e469`.
2. `scripts/wp7_capture_gpu.py` — restored from the same tracked commit. Canonical LF blob SHA-256 `487b7e6616f6b33608a78b43055e785638e4aadfb45847c77ed76401d0f8a11f`.

The CRLF Windows working-tree copies from two independent older checkouts were content-equal to the Git blobs after line-ending normalization. **No file from the dirty modern checkout was used as restoration source.** Restored dependencies do not authorize training or run any PPO.

The missing-file recovery exposed an additional *denied-admission report* failure: `audit_train_tracks` tried validating **all** manifest tracks, including absent HELD-OUT geometry. A one-call-site change in `f1rl/research/wp7_track_feasibility.py` now passes `validate_asset_ids=TRAIN_IDS` to the original loader. This still verifies every historical TRAIN-4 asset and retains HELD-OUT metadata without accessing geometry. The default TRAIN-4 identities/manifest, Physics, reward, observation, race, collision and PPO source remain unchanged.

Test-first independent regression: the two new tests proved the old behavior RED, then passed GREEN. Original two failing tests now **2/2 PASS**. The expanded historical compatibility group (including the two new tests) reports **26 PASS, 0 FAIL, 0 ERROR**. The updated Step4/Physics-v4/source-bound suite reports **194 PASS, 0 FAIL, 0 ERROR**. The two groups overlap by two tests and should not be naively added as unique coverage.

## Source lock

`configs/research/wp7-train5-paired-v2-registration.json` was regenerated to bind **44** actual source/config/asset/model identities and the unchanged **4** external Step3 dual-turn Physics-v4 receipts. This extends the earlier 41-file source lock to include both restored historical files and the now-covered TRAIN-only audit implementation. The original 41-file registration is archived in external evidence as `step4-initial-41-file-proposal-snapshot.json`; no old scientific result or model hash was overwritten.

This registration remains `draft_not_authorized`, with `training_permission=false`, `smoke_permission=false`, `heldout_permission=false`, `global_seed_registry_receipt=null`, and `owner_approval=null`. Existing original 4-track training/split and 12 frozen checkpoint archives are not replaced.

## Seed history / limitations

An independent bounded scan of **67 F1-RL historical folders** on the accessible D: drive examined **22,772 text, source, config and evidence/log files** totaling **376,233,204 bytes**. It found **zero** use occurrences of the 18 proposed TRAIN-5 seed integers in this inspected material and **zero read errors**. However, **24 large text files were skipped**, and scanned folders exclude binary model data, compressed archives, Git pack/dangling or deleted history, other drives/machines, cloud/Colab, collaborators and historical records held elsewhere.

**Verdict: LOCAL-SOURCE-NO-COLLISION. Never equate this with Global Seed Registry verified/unused proof.** No seed was activated, allocated to a run or used for training. Full owner-signed seed registry cross-check is still required.

## CI and source receipt boundaries

The exact local subset regressions are green, but **this is not GitHub CI of a committed release candidate**. The branch `handoff/physics-v4-khanh-bang-20261106` remains on original HEAD `1de279c37e62a27531d661480d20fa6e095c4e84` with deliberate WIP source changes. No commit or push was made under the draft handoff's prohibition; consequently no remote workflow can validate these uncommitted files. A full project-wide pytest suite has not been represented as passing.

The preexisting evidence archive remains separate under `D:\F1-RL-PHYSICS-V4-EVIDENCE-20261106`; the prior 20 frozen source inputs and 30 archived evidence checksums must stay unchanged. The older dirty project checkout is not to be reset or modified.

## Remaining release blockers

1. Inspect the 24 skipped large files and relevant compressed/historical backup archives with a bounded, auditable method. Verify other machines, old offline training runs, Colab and collaborators, then owner-sign the Global Seed Registry. A local no-match scan alone is not sufficient.
2. Produce a reviewed **clean committed release candidate** and run the required GitHub Actions workflow/CI on its exact commit, plus source SHA lock refresh if any source changes. This needs separate authorization to commit/push given the draft no-push contract.
3. Complete independent scientific protocol review for full TRAIN-5 PPO training (seeds, fine-tune selection and hyperparameters, evaluation/abort rules), then owner approval. Do not repurpose the draft's fail-closed boolean fields as a shortcut.
4. Obtain **separate approval** for deterministic six-case frozen PPO turn smoke. The tests and source/seed audit performed here **did not run any smoke episode**.
