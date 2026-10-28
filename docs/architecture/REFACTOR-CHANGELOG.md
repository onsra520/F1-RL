# Refactor changelog

- Baseline `a909a25`: engineering-accepted consolidation.
- `b86379f`: source-backed architecture map, baseline file/hash inventory and migration decisions. No application behavior changed.
- Package migration was investigated in an independent detached clone and temporarily in the architecture worktree. The unaccepted batch was reverted from the worktree; an experimental patch remains in the external archive for investigation only. It is not a completed migration and must not be applied without reviewing scientific source-bound contracts and regression coverage.
