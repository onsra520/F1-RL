# WP7 checkpoint observer plumbing — training-free verification

**Recovery checkpoint:** 2026-10-11. This is preparatory infrastructure, not a 200k scientific pilot.

`TrainingRunner` accepts an optional `checkpoint_observer: Callable[[Path, int], None]`. The existing API and behavior are identical when this callback is omitted.

- The observer executes only **after `PPOModel.save()` successfully returns** and the checkpoint has been appended to the in-progress artifact list.
- The hook receives `(archive_path, actual_timesteps)`. Crucially, *actual* timesteps may differ from configured thresholds when PPO/vectorized stepping overshoots a milestone. A future WP7 observer must explicitly validate and record actual-vs-requested milestones; it must never report an exact 25k result if the archive was saved at 25,008.
- An observer exception aborts training, flows into the runner's existing exception recovery, sets `state.status=failed`, preserves the already-saved checkpoint, and prevents a misleading successful final model.
- Existing checkpoint scheduling, saving, worker cleanup and training APIs are otherwise unchanged.
- No checkpoint observer is active in the recovery candidate as of this commit. No PPO training, model loading, held-out rollout, or final WP8 seed was used.

TDD verification: `tests/learning/training/test_checkpoint_observer_reconstruction.py` uses mocked PPO and vector environments. It tests ordered 25k, 50k, 75k observer notifications from existing saved archives; observer failure preserving failure state; rejected invalid callback; and unchanged no-observer behavior. The runner's pre-existing unit suite is also rerun.

**Next distinct scientific gates:** build a dedicated observer that verifies the recovery YAML + TRAIN asset content-v2 fingerprints and writes atomic checkpoint evaluation reports; verify all 25k milestone coverage plus 200k final evaluation and independent 0/0 versus 1/1 pilot seeds; make artifact storage durable outside ignored worktree files. The candidate preflight still reports `review_required`; no pilot is authorized by this infrastructure change alone.
