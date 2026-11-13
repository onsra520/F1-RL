# WP3 Issue #51 — R3 seven hard-stop failures repaired, clean replays, and R4 prepared

**Engineering replay result: 7/7 VALID_EPISODE, zero INVALID_EPISODE** on clean source `929f05fa170e8d9f10ce3d150a557d7408ac0fa2`. This evidence is not independent scientific certification and not a completed new 5,760-episode Evaluation.

## Source, model and failure provenance

- Source branch: `fix/wp3-seven-hardstop-edge-contact-20261026`; source commit **`929f05fa170e8d9f10ce3d150a557d7408ac0fa2`**, worktree clean.
- Historical 7 invalid episodes from immutable `WP3-OWNER-TRAINDEV-20261025-FULL-03` (source `56e229f`): C-seed0 right-turn randomized e29; B-seed1 slalom nominal e22; B-seed1 right-turn randomized e58; C-seed1 right-turn nominal e14; C-seed1 right-turn randomized e4; A-seed2 right-turn nominal e39; B-seed2 slalom randomized e26.
- Same 12 frozen PPO checkpoints and exact paired environment/policy seed fields, source and checkpoint digests bound by `REPLAY-MANIFEST.json`.
- No previous evaluation, HELD-OUT source, checkpoint ZIP or roster rewritten.

## Physics correction

The failures occurred when the finite tire footprint was OUTSIDE a road polygon but already below the vertical suspension bump-stop plane, then entered the polygon footprint. The previous solver mistakenly tried to certify a vertical bump-stop crossing that had not occurred, failing closed.

New **development-only finite-tire sidewall edge entry**:
- Proves a *unique* road polygon edge and strictly entering tire footprint, while the wheel is already below the unchanged 0.12 m hard-stop boundary;
- Checks the incoming tire-sidewall normal velocity and computes the real rigid horizontal contact impulse, angular impulse, non-increasing kinetic energy and accepted post-impact vehicle state;
- Bisects full RK4 trial leaves to find the **first valid last-outside accepted prefix**, with source-bound contact time and ≤0.1 ns geometric bracket;
- Preserves the exact terminal prefix, fail-closed rules for ambiguous corners, fake edge contact, nonapproaching velocities or forged work, and distinct crash source registration and independent replay verification;
- Never labels an invalid sample as a finish or grants scientific physics reviewer approval.

## Verification
- **7/7 frozen R3 invalid episodes replayed as valid physical_crash**, 0 invalid; replay checks clean source commit before and after. Raw `REPLAY-RESULTS.jsonl` SHA256: `d8054bdf95a1293d94d033e25a1ba0f42f79065ee053532396a80f1ecdd0950f`.
- Focused Physics, crash, evaluator, admissions and anti-forgery regression **160/160 PASS** after commit on clean source.
- Whole-repository `pytest` started on the new source with separate stdout/exit receipt, but a final PASS is NOT claimed until it completes.

## New TRAIN-development Evaluation R4
- NEW run `WP3-OWNER-TRAINDEV-20261026-FULL-04`
- Source `929f05fa170e8d9f10ce3d150a557d7408ac0fa2`
- New host root `D:\F1-RL-WP3-OWNER-FULL-TRAINDEV-20261026-R4`
- 12 frozen PPO final ZIPs, 96 cells × 60 paired original seeds = **5,760 TRAIN-only episode slots**.
- Manifest digest `fd23fba13324691fb700e6c01ca20917b7b4bf6cfa11490a923f915959fc73dc`.
- Evidence package contains the exact new RUN-MANIFEST and RUN-STATE-INITIAL, both registered while **0 episodes had started**.
- A source-external fail-closed controller `WP3-929F05F-GUARDED-LAUNCH-20261026.ps1` has been started. It verifies the complete full-suite zero exit receipt, exact clean source, manifest hash, empty destination and absence of duplicate workers **before** attempting any TRAIN-development workers.
- Scientific gate remains **BLOCKED_PENDING_INDEPENDENT_REVIEW**. Only a future complete 96/96 valid cell run and independent reviewer signoff may change the scientific status.

The historical R3 incomplete result remains `INCOMPLETE_RANKING_WITHHELD`. No claim of R4 execution or completion is made by this initial evidence.
