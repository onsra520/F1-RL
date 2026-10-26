# WP3 Issue #51 — certified full-quaternion roll-domain transition (26 Oct 2026)

## Source frozen
Branch `fix/wp3-tilt-event-20261026` — commit **`13b1d8105d14e7492a140a850c43f22e578fd809`**, clean, pushed.
Source is based on approved Physics v4 post seven hard-stop corrections (`929f05fa170e8d9f10ce3d150a557d7408ac0fa2`).

## Original R4 invalid replay
- Historical immutable R4 run: `WP3-OWNER-TRAINDEV-20261026-FULL-04`, 95/96 cells, 5,710 valid episodes, one invalid and 49 unattempted, ranking WITHHELD.
- Exact R4 case: `C-seed1 / right-turn-v1 / randomized_train / episode 10`; original environment seed `3886169428`, policy seed `1971191095`. Frozen checkpoint hashes and roster digest were re-verified. Original R4 cell SHA recorded in the event artifact.
- On unchanged source `929f05f`, the original `SuspensionResolutionError` is reproducible: roll `0.704367412 > 0.7` rad.
- On new clean source `13b1d81`, the same one case replays as **VALID_EPISODE / hard_route_violation**, episode time `14.1 s`. The game is NOT marked a successful finish or fabricated crash. No retraining or held-out.
- The dedicated source-instrumented production replay observed **exactly one** signed full-quaternion roll-boundary event in an already accepted RK4 leaf:
  - start roll `0.6916810232428126`, end roll `0.7043676124648689`, unchanged limit `0.7`;
  - first proven clear `0.004098063427954913 s`, first witnessed crossing `0.00409806352108717 s` in the leaf (width ~`9.31e-11 s`);
  - independently reverified from accepted before/after quaternions, with linear momentum, angular momentum and mechanical energy residuals within unchanged tolerances.
- Underlying finite-road OBB SAT at old rejected stage remained separated from body by a minimum `0.142413...` m. We **did not invent** a road impact, collision normal, contact impulse or kinetic loss.

## Nature of the correction
Physics v4 already uses full world-space quaternion + angular momentum equations and a 3D finite-tyre contact model. The `max_abs_tilt_rad = 0.7` policy comes from legacy small-angle suspension coordinates; it remains unchanged, and all legacy mode checks remain strict. Only v4 large **roll** is admitted *through the full-quaternion equations*, with new short accepted-RK4 roll-boundary proof. Pitch, linear speed and angular speed bounds are unchanged. The exact analytical derivative of roll along SLERP establishes monotonicity over the full interval by a conservative Lipschitz bound; the first crossing bracket is refined to <=`1e-10 s`. The runtime Physics validator recomputes each signed event from accepted source poses; road/wheel impact validators also verify previous signed roll leaves. All source RK4 momentum/work/energy and bounded body/road sweeps remain intact.

No signal in this evidence certifies general scientific realism of rollover behaviour or grants an independent physics-review signature. This is engineering development evidence only.

## Tests and release
- Focused Physics, evaluator, contact, and anti-forgery regression: **178 passed, zero failures** before source commit.
- Original case replay **PASS** after source clean Git commit, with dedicated stamped SHA receipt and source-integrated event proof.
- Full repository regression on exact clean commit is being executed separately. Its final exit receipt, not partial pytest progress, determines its official PASS/FAIL state.
- **DO NOT LAUNCH OR PREPARE ANOTHER FULL EVAL** from this evidence; explicitly held per owner request until owner asks to resume.
- Historical R3/R4 results, checkpoint ZIPs, TRAIN roster and sealed HELD-OUT have not been rewritten.

Raw replay outcome, signed roll event, frozen seed provenance, and their SHA checksums are stored alongside this README.
