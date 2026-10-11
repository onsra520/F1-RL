# WP7 recovery — eight-milestone, TRAIN-only mock evidence pipeline

**As of 2026-10-11: mocked lifecycle verified, no PPO pilot executed.**

## Scope

`TrainingRunner` has two optional model observer hooks. The checkpoint observer evaluates/saves evidence after each intermediate checkpoint, and the final-model observer executes after the final archive is saved but **before** `state.json` is marked completed. An observer exception propagates and marks the run failed; an invalid final model is not published as a successful result.

`PilotMilestoneRecorder` implements this contract for the preregistered WP7 recovery plan. It remains **blocked for real training** while `build_pilot_preflight().review_blockers` is nonempty or the Git worktree is dirty. In **explicit mock mode**, a test evaluator must be injected. Mock-mode reports **always** declare `scientific_evidence: false`.

It binds exactly four TRAIN assets and the candidate/profile/base-experiment/manifest fingerprints. Each observer checks source integrity and all previously published model/report SHA256 hashes before accepting a new milestone. A missing or mutated archive, incorrect track ID, returned HELD-OUT evidence, spoofed checkpoint SHA, skipped milestone, unchanged checkpoint artifacts, or an exception aborts processing.

## Artifact layout after a mocked successful run

```text
<run-directory>/
  state.json
  checkpoints/ppo-multitrack-...-step25000.zip
  checkpoints/...
  ppo-multitrack-...-final-model.zip
  evaluations/
    milestone-025000.json
    milestone-050000.json
    milestone-075000.json
    milestone-100000.json
    milestone-125000.json
    milestone-150000.json
    milestone-175000.json
    milestone-200000.json
    manifest.json
```

Each milestone report records separately the *nominal target* (25k, 50k, ..., 200k) and the PPO **actual** saved timestep (e.g. actual 200704 for a 200000 target), plus archive SHA256, Git revision, candidate file hashes, evaluation results and `heldout_rollouts: 0`. All report hashes appear in the manifest. JSON files are written through a same-directory temp file then `os.replace`, and existing report names are rejected by the single-owner writer.

To verify the *mock evidence* from a copied run directory, without loading a policy or starting any simulation:

```powershell
cd D:\Projects\.worktrees\F1-RL\wp7-rebuild
.\.venv\Scripts\python.exe scripts\wp7_verify_pilot_evidence.py "<run-directory>"
```

**Passing this verification only proves checksum consistency, not model skill, scientific provenance or durability against loss of the only physical disk.** The actual model files and report directory are still necessary; Git commits do **not** contain model checkpoint weights.

## Mock end-to-end tests

`tests/learning/training/test_pilot_milestone_recorder_reconstruction.py` drives the original runner through 200k *mocked timesteps* using `MockPPOModel` and `MockVecEnv` and a fake TRAIN evaluator. The tests check:
- Seven intermediate checkpoints plus one final 200k report, report-to-archive and report-to-manifest hash binding.
- Actual 200704 vs nominal 200000 preservation.
- Mid-run evaluator failure, under-200k final failure, archive mutation during evaluation and previously published report/checkpoint tampering.
- Post-copy verifier detecting corrupted model/archive/report bytes.
- Real mode still blocked on explicit scientific `review_required` gates.
- Existing training-runner API/tests unchanged without observers.

## Still open before a real 200k run

1. **Physical swept-volume validation** on slalom and right-turn including vehicle footprint and checkpoint collision logic.
2. Review reward shortcut exploit resistance and precise C2 semantic freeze policy.
3. A reviewed production wiring layer that loads a real PPO checkpoint, runs reproducible TRAIN-only per-track episodes, and stores additional runtime/provenance state. The current fake evaluator must not be treated as that implementation.
4. Reliable artifact backup off the newly configured machine/SSD, atomic filesystem behavior after power interruption, and evidence restoration checks.
5. Fresh independent calibration/confirmation seeds `(0,0)` and `(1,1)`, plus all eight evaluation points. Previous successful pilot claims cannot substitute for lost weights/reports.

No PPO training, learned HELD-OUT evaluation, final replicate seeds, WP8, or merge to `main` occurred in these mock tests.
