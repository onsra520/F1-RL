# Issue #51: exact frozen-seed replays of 29 failed WP3 episodes (25 Oct 2026)

**Engineering replay only — FAIL (19/29 valid, 10/29 still invalid).** Not an official scientific Evaluation and **NOT a successful Physics v4 completion**.

## Original frozen input
- Original evaluation run `WP3-OWNER-TRAINDEV-20261024-FULL-01` contained **29** invalid episode attempts among 4,502 started slots, 67 complete cells and 1,258 unattempted slots. The original 96-cell roster, twelve frozen PPO ZIP checkpoints and paired RNG seeds were read-only; no redraw or HELD-OUT access.
- Original failure classes: 18 `PhysicsTransitionError`, 10 `SuspensionResolutionError`, 1 `GroundedBodyCollisionUnscored`.
- Scratch replay ID `WP3-WIP-29-INVALID-REPLAY-20261025-R2` and output: `D:\F1-RL-WP3-ISSUE51-29-INVALID-REPLAY-20261025-R2`.
- Source at replay: Git HEAD `bbefa1b2e0261b34cd826f896be3cfb838ecdc90` **plus UNCOMMITTED, per-file SHA-bound engineering fixes**. Inspect `REPLAY-MANIFEST.json` for every modified file. The runner checked that fingerprint **both before and after** all 29 replays and rejected drift.
- Frozen roster/source/checkpoint integrity checked. No new 5,760-episode run was started.

## Results
- **19 VALID_EPISODE** on replay, including genuine failed driving outcomes (`physical_crash`, `step_limit` etc.). They are not fabricated finishes.
- **10 INVALID_EPISODE**, all `SuspensionResolutionError`:
  - **4** new wheel-support acquisitions caused an energy jump and exhausted microscopic RK4 refinement. Example original C-seed0 / right-turn-v1 / randomized_train / episode 18, incoming wheel contact support raises modeled spring energy from 470.616 J to 1004.057 J over 5.96 ns.
  - **6** exceed the registered pitch/roll domain `max_abs_tilt_rad=0.2` (angles 0.201–0.204 rad), not an independently certified contact/crash terminal. Merely raising tilt tolerance or reclassifying a solver exception as a crash is not evidence of physical correctness.
- Exact per-episode JSONL and model/track/seed fields are in `REPLAY-RESULTS.jsonl`. The SHA-256 checksum of this log is `3c91e4aa1dc295eeddeaa60d3c1de69124ec195a602614a23ee87848f2aa3b74`.

## Source status / next gate
The source edits fix one-ULP accepted road-crash quaternion reconstruction and support exact certified edge-on-edge obstacles in engineering tests. They do **not** implement certified finite-wheel road-edge recontact impulse/work at existing compression, nor independently certified body–road response beyond the 0.2-rad orientation domain. **No released source and no new full Evaluation are authorized until all remaining cases replay without physics exceptions**, regression is green, and source-bound review is complete.

The complete `pytest` suite was started separately but at the time this evidence was prepared had observable failures; no all-green regression claim is made. The prior owner-authorized run remains `INCOMPLETE_RANKING_WITHHELD`.
