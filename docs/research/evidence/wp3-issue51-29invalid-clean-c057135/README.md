# WP3 Issue #51 — all 29 original invalid episodes replayed on clean source

**Engineering regression PASS: 29/29 valid episodes, 0 invalid.** This is not the new full 5,760-episode Evaluation nor independent scientific acceptance.

Source branch `fix/wp3-eval-physical-completion`, locked source SHA `c057135a329af59e07d9823a7bedf4b2b06f8682`, with **zero dirty source files** before and after the replay. The 29 original failed episode identities, 12 frozen model ZIP hashes and original paired environment/policy RNG seeds are bound in `REPLAY-MANIFEST.json`.

Original invalid breakdown: `SuspensionResolutionError` 10, `PhysicsTransitionError` 18, `GroundedBodyCollisionUnscored` 1. All 29 previously invalid episode slots now execute with a valid driving terminal, including correctly scored crashes and hard-route violations rather than fabricated finishes.

The 10 suspension cases include four original positive spring-energy discontinuities from hard road-edge point support, and six exceeded tilt-domain cases. Fix uses finite tyre contact footprint and conservative edge potential gradient and certified contact/terminal handling; preserving a separate new experimental Physics v4 source. It does not constitute third-party physical calibration.

- Focused integration/regression: **151/151 PASS** on the clean commit.
- Source unchanged during replay; no retraining, no HELD-OUT, original run and evidence untouched.
- Per-episode results: `REPLAY-RESULTS.jsonl`; raw SHA-256 `745ba524b687f6ec22ffa295a4e7f9edd14850785875820747097d2ef16ac419`.
- Manifest, per-episode replay, and summary supplied as original byte-preserved JSON/JSONL.
- Full repository pytest **not yet verified green** on this new commit. Full 5,760 development Evaluation still requires a separate new run ID and 96/96 valid cells; independent scientific review remains separate.
